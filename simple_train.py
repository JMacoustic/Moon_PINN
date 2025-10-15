import torch
from src.models.simple_pinn import *
import os
from dataclasses import dataclass
from typing import Dict
from pathlib import Path
from src.utils.animate import animate

# ------------------ Helper ------------------

def _infer_initial_thickness_from_aux(aux):
    """t0 ≈ 2*s from meta if present; else estimate from corners."""
    if 'meta' in aux and 's' in aux['meta']:
        return float(2.0 * aux['meta']['s'])
    # fallback: estimate from vertices & lattice points
    P = np.asarray(aux['lattice_points'], dtype=float)
    V = np.asarray(aux['verts'], dtype=float).reshape(-1,2)
    # choose a center node and find nearest vertex distance ≈ sqrt(2)*s
    ny, nx, _ = P.shape
    j = min(1, ny-2) if ny>=3 else 0
    i = min(1, nx-2) if nx>=3 else 0
    cx, cy = P[j,i]
    d2 = (V[:,0]-cx)**2 + (V[:,1]-cy)**2
    s = float(np.sqrt(d2.min())/np.sqrt(2.0))
    return 2.0*s

def _extract_initial_px_py_xoff(aux):
    P = np.asarray(aux['lattice_points'], dtype=float)
    ny, nx, _ = P.shape
    px = float(np.mean([P[0,i+1,0]-P[0,i,0] for i in range(nx-1)]))
    py = float(np.mean([P[j+1,0,1]-P[j,0,1] for j in range(ny-1)]) if ny>1 else 1.0)
    # diffs along a row alternate (px-2xoff, px+2xoff, ...)
    if nx >= 3:
        d0 = P[0,1,0]-P[0,0,0]
        d1 = P[0,2,0]-P[0,1,0]
        xoff = 0.25*abs(d1-d0)
    else:
        xoff = 0.0
    return px, py, xoff

def thickness_from_constraint(C, px, py, xoff, eps=1e-9):
    denom = (py + 0.5*px + xoff)
    denom = float(max(denom, eps))
    return float(C / denom)

def _refresh_mesh_inplace(mesh, aux):
    """Update mesh geometry from aux (verts, bcIds, etc.) without changing connectivity."""
    V = np.asarray(aux['verts'], dtype=float).reshape(-1,2)
    mesh.verts = torch.tensor(V, dtype=torch.float32, device=mesh.verts.device if hasattr(mesh, 'verts') else 'cpu')
    # If your Mesh caches AABBs/areas/triangles, call its internal rebuild hooks here:
    if hasattr(mesh, "rebuild_accel"):
      mesh.rebuild_accel()
    if hasattr(mesh, "top_ids") and hasattr(mesh, "bottom_ids"):
      # recompute top/bottom from aux['bcIds'] only for 'top' (max y); bottom = min y
      y = mesh.verts[:,1].detach().cpu().numpy()
      y_max, y_min = float(y.max()), float(y.min())
      mesh.top_ids = torch.tensor(np.where(np.abs(y - y_max) < 1e-9)[0], dtype=torch.long, device=mesh.verts.device)
      mesh.bottom_ids = torch.tensor(np.where(np.abs(y - y_min) < 1e-9)[0], dtype=torch.long, device=mesh.verts.device)

# ------------------ Config ------------------
@dataclass
class DesignState:
    px: float
    py: float
    xoff: float
    C: float      # constraint constant

@dataclass
class Material:
    E: float = 1.0e1       # Young's modulus (arbitrary units)
    nu: float = 0.30       # Poisson's ratio
    rho: float = 1.0       # density
    plane_stress: bool = True

@dataclass
class TrainCfg:
    name: str = "251001_testrun_v0"
    T: float = 10.0        # total time
    f_top: float = 0.5     # Hz for top displacement
    V0: float = 0.05       # amplitude of top vertical displacement
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 1000
    lr: float = 1e-3
    pde_batch: int = 8192
    bc_batch: int = 2048
    ic_batch: int = 2048
    
    # loss weights
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 2.0
    w_vib: float = 1.0        # weight for bottom vibration loss
    vib_steps: int = 25       # time steps used to measure vibration

# -------------- Training loop --------------
@torch.no_grad()
def _stats(name, tensor):
    return f"{name}: mean={tensor.mean().item():.3e} rms={tensor.pow(2).mean().sqrt().item():.3e}"

def train_pinn(aux_dict: Dict, mat=Material(), cfg=TrainCfg()):
    os.makedirs("outputs", exist_ok=True)

    # ---- init design vars from the provided aux ----
    px0, py0, xoff0 = _extract_initial_px_py_xoff(aux_dict)
    t0 = _infer_initial_thickness_from_aux(aux_dict)
    C_const = (py0 + 0.5*px0 + xoff0) * t0
    design = DesignState(px=px0, py=py0, xoff=xoff0, C=C_const)

    mesh = Mesh(aux_dict)
    device = torch.device(cfg.device)

    pinn = ElastodynamicsPINN(mesh, mat, cfg).to(device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)

    # optional: separate lr for geometry (outer loop)
    geom_lr = getattr(cfg, "geom_lr", 1e-3)
    fd_eps  = getattr(cfg, "geom_fd_eps", 1e-3)  # relative perturbation for finite-diff
    geom_every = getattr(cfg, "geom_every", 1)   # update geometry every N epochs

    # -------- samplers (unchanged) --------
    def sample_pde_batch():
        P = mesh.sample_interior(cfg.pde_batch).to(device)
        t = (cfg.T * torch.rand(cfg.pde_batch,1, device=device))
        return P[:,0].view(-1), P[:,1].view(-1), t.view(-1)

    def sample_bc_bottom():
        P = mesh.sample_on_nodes(mesh.bottom_ids, cfg.bc_batch).to(device)
        if P.numel()==0:
            return (torch.empty(0, device=device),)*3
        t = (cfg.T * torch.rand(P.shape[0],1, device=device))
        return P[:,0], P[:,1], t.view(-1)

    def sample_bc_top():
        P = mesh.sample_on_nodes(mesh.top_ids, cfg.bc_batch).to(device)
        if P.numel()==0:
            return (torch.empty(0, device=device),)*3
        t = (cfg.T * torch.rand(P.shape[0],1, device=device))
        return P[:,0], P[:,1], t.view(-1)

    def sample_ic_zero():
        P = mesh.sample_interior(cfg.ic_batch).to(device)
        t0 = torch.zeros(P.shape[0], device=device)
        return P[:,0], P[:,1], t0

    mse = nn.MSELoss()

    # -------- training --------
    for ep in range(1, cfg.epochs+1):
        # ---------------- PINN step ----------------
        opt.zero_grad(set_to_none=True)

        x, y, t = sample_pde_batch()
        rx, ry = pinn.pde_residual(x, y, t)
        loss_pde = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry))

        xb, yb, tb = sample_bc_bottom()
        loss_bc_bot = torch.tensor(0.0, device=device)
        if xb.numel()>0:
            ub, vb = pinn.bc_bottom_clamp(xb, yb, tb)
            loss_bc_bot = mse(ub, torch.zeros_like(ub)) + mse(vb, torch.zeros_like(vb))

        xt, yt, tt = sample_bc_top()
        loss_bc_top = torch.tensor(0.0, device=device)
        if xt.numel()>0:
            ru, rv = pinn.bc_top_disp(xt, yt, tt)
            loss_bc_top = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))

        xi, yi, ti = sample_ic_zero()
        ui0, vi0, uvt0, vvt0 = pinn.ic_zero(xi, yi, ti)
        loss_ic = (mse(ui0, torch.zeros_like(ui0)) + mse(vi0, torch.zeros_like(vi0)) +
                   mse(uvt0, torch.zeros_like(uvt0)) + mse(vvt0, torch.zeros_like(vvt0)))
        
        loss_vib = pinn.bottom_vibration_loss(time_steps=getattr(cfg, "vib_steps", 25))

        loss = cfg.w_pde*loss_pde + cfg.w_bc*(loss_bc_top) + cfg.w_ic*loss_ic + getattr(cfg, "w_vib", 1.0)*loss_vib
        loss.backward()
        opt.step()
        sched.step()

        # ---------------- outer geometry step ----------------
        if (ep % geom_every) == 0:
            # choose objective for design. If you have bottom_amplitude_loss, use that:
            with torch.enable_grad():
                try:
                    design_obj_base = float(pinn.bottom_vibration_loss(time_steps=getattr(cfg, "vib_steps", 25)))
                except Exception:
                    # fallback: recompute a quick PDE residual proxy fresh
                    x_, y_, t_ = sample_pde_batch()
                    rx_, ry_ = pinn.pde_residual(x_, y_, t_)
                    design_obj_base = float((rx_.pow(2).mean() + ry_.pow(2).mean()).detach().cpu())

            def eval_with(des):
                # mutate aux and refresh mesh
                t_new = thickness_from_constraint(design.C, des.px, des.py, des.xoff)
                mutate_aux_inplace(aux_dict,
                                pitch=(des.px, des.py),
                                x_offset=des.xoff,
                                thickness=t_new,
                                keep_meta=True)
                _refresh_mesh_inplace(mesh, aux_dict)
                if hasattr(pinn, "set_mesh"):
                    pinn.set_mesh(mesh)

                # Prefer the vib metric if available
                if hasattr(pinn, "bottom_vibration_loss"):
                    with torch.no_grad():  # safe: this path doesn’t need gradients
                        return float(pinn.bottom_vibration_loss(time_steps=getattr(cfg, "vib_steps", 25)))

                # Fallback proxy: needs gradients -> DO NOT use no_grad here
                x, y, t = sample_pde_batch()
                rx, ry = pinn.pde_residual(x, y, t)  # requires grad-enabled inputs
                return float((rx.pow(2).mean() + ry.pow(2).mean()).detach().cpu())

            # finite-difference gradients (relative eps)
            base = DesignState(design.px, design.py, design.xoff, design.C)
            g_px = 0.0; g_py = 0.0; g_x = 0.0

            for name in ["px","py","xoff"]:
                des_p = DesignState(base.px, base.py, base.xoff, base.C)
                val   = getattr(base, name)
                step  = max(1e-6, abs(val)*fd_eps)
                setattr(des_p, name, val + step)
                Jp = eval_with(des_p)

                des_m = DesignState(base.px, base.py, base.xoff, base.C)
                setattr(des_m, name, max(1e-9, val - step))  # keep positive
                Jm = eval_with(des_m)

                g = (Jp - Jm) / (2.0*step)
                if name == "px":   g_px = g
                if name == "py":   g_py = g
                if name == "xoff": g_x  = g

            # gradient step (minimize)
            design.px  = max(1e-6, design.px  - geom_lr * g_px)
            design.py  = max(1e-6, design.py  - geom_lr * g_py)
            design.xoff= max(0.0,  design.xoff- geom_lr * g_x)  # keep non-negative

            # final commit after update
            t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            mutate_aux_inplace(aux_dict,
                               pitch=(design.px, design.py),
                               x_offset=design.xoff,
                               thickness=t_new,
                               keep_meta=True)
            _refresh_mesh_inplace(mesh, aux_dict)
            if hasattr(pinn, "set_mesh"):
                pinn.set_mesh(mesh)

        # ---------------- logs + snapshots ----------------
        if ep % 100 == 0 or ep == 1:
            t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            print(f"[Ep {ep:05d}] loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                f"bcT={loss_bc_top.item():.3e} | ic={loss_ic.item():.3e} | vib={loss_vib.item():.3e} | "
                f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_now:.4f}")

        # (your snapshot saving block unchanged) ...
        outdir = Path("outputs/data"); outdir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            times = np.linspace(0, cfg.T, 21)
            verts = mesh.verts.to(device)
            x, y = verts[:, 0], verts[:, 1]
            for k, tk in enumerate(times):
                t_k = torch.full_like(x, fill_value=float(tk))
                u, v = pinn.forward(x, y, t_k)
                disp = torch.stack([u, v], dim=1).cpu().numpy()
                np.savez_compressed(outdir / f"{cfg.name}_{k:04d}.npz", t=float(tk), disp=disp)

    return pinn


# ----------------- Example usage -----------------
if __name__ == "__main__":
    aux = generate_auxetic()
    pinn = train_pinn(aux, mat=Material(E=20.0, nu=0.3, rho=1.0, plane_stress=True), cfg=TrainCfg())
    animate(cfg=TrainCfg())
