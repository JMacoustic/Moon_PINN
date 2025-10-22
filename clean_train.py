import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from src.utils.animate import animate
from src.utils.smooth_auxetic import *
from src.models.simple_pinn import *

# ---- assumes your optimized Aux class (with adjust_geometry) is imported ----
# from auxetic import Aux
# assumes your ElastodynamicsPINN works with an object exposing:
#   - verts_torch (nv,2), tris_torch (nt,3)
#   - sample_interior(n), sample_on_nodes(ids,n)
#   - bottom_ids, top_ids
# and has: pde_residual, bc_bottom_clamp, bc_top_disp, ic_zero, bottom_vibration_loss
# Optionally pinn.set_mesh(mesh_like) to notify about geometry updates.


# ------------------ Config ------------------
@dataclass
class DesignState:
    px: float
    py: float
    xoff: float
    C: float      # constraint constant


@dataclass
class Material:
    E: float = 1.0e1
    nu: float = 0.30
    rho: float = 1.0
    plane_stress: bool = True


@dataclass
class TrainCfg:
    name: str = "251001_testrun_v0"
    T: float = 10.0
    f_top: float = 0.5
    V0: float = 0.05
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
    w_vib: float = 1.0
    vib_steps: int = 25

    # simple outer-loop geometry hyperparams
    geom_lr: float = 1000
    geom_fd_eps: float = 1e-3
    geom_every: int = 1


@torch.no_grad()
def _stats(name, tensor):
    return f"{name}: mean={tensor.mean().item():.3e} rms={tensor.pow(2).mean().sqrt().item():.3e}"


def thickness_from_constraint(C: float, px: float, py: float, xoff: float) -> float:
    # same formula you used to define C_const initially
    denom = (py + 0.5 * px + xoff)
    return max(1e-9, C / max(1e-9, denom))


def train_pinn(aux: "Aux", mat=Material(), cfg=TrainCfg()):
    """
    Train loop using a single Aux mesh object.
    Geometry updates use Aux.adjust_geometry for speed (no rebuild of connectivity).
    """
    os.makedirs("outputs", exist_ok=True)
    device = torch.device(cfg.device)

    # ---- init design state from aux ----
    px0, py0 = aux.pitch
    xoff0 = aux.x_offset
    t0 = float(0.5 * aux.thickness if aux.joint_radius is None else aux.thickness)  # not used; just here if you track
    # keep your original constraint definition:
    t_init = aux.thickness
    C_const = (py0 + 0.5 * px0 + xoff0) * t_init
    design = DesignState(px=px0, py=py0, xoff=xoff0, C=C_const)

    # ---- build PINN ----
    pinn = ElastodynamicsPINN(aux, mat, cfg).to(device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=1e-6)
    mse = nn.MSELoss()

    # -------- samplers --------
    def sample_pde_batch():
        P = aux.sample_interior(cfg.pde_batch).to(device)
        t = (cfg.T * torch.rand(cfg.pde_batch, 1, device=device))
        return P[:, 0], P[:, 1], t.view(-1)

    def sample_bc(ids):
        P = aux.sample_on_nodes(ids, cfg.bc_batch).to(device)
        if P.numel() == 0:
            z = torch.empty(0, device=device)
            return z, z, z
        t = (cfg.T * torch.rand(P.shape[0], 1, device=device))
        return P[:, 0], P[:, 1], t.view(-1)

    def sample_ic_zero():
        P = aux.sample_interior(cfg.ic_batch).to(device)
        t0 = torch.zeros(P.shape[0], device=device)
        return P[:, 0], P[:, 1], t0

    # -------- helper: evaluate objective under temporary geometry --------
    def evaluate_objective(px, py, xoff) -> float:
        # save current
        px_c, py_c = aux.pitch
        x_c = aux.x_offset
        t_c = aux.thickness

        # set new geometry respecting constraint via thickness
        t_new = thickness_from_constraint(design.C, px, py, xoff)
        aux.adjust_geometry((px, py), xoff, t_new)
        pinn.set_mesh(aux)

        # objective: prefer vibration metric
        with torch.no_grad():
            J = float(pinn.bottom_vibration_loss(time_steps=cfg.vib_steps))


        # restore previous geometry
        aux.adjust_geometry((px_c, py_c), x_c, t_c)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)
        return J

    # ---------------- training ----------------
    for ep in range(1, cfg.epochs + 1):
        # ----- PINN step -----
        opt.zero_grad(set_to_none=True)

        x, y, t = sample_pde_batch()
        rx, ry = pinn.pde_residual(x, y, t)
        loss_pde = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry))

        xb, yb, tb = sample_bc(aux.bottom_ids)
        loss_bc_bot = torch.tensor(0.0, device=device)
        if xb.numel() > 0:
            ub, vb = pinn.bc_bottom_clamp(xb, yb, tb)
            loss_bc_bot = mse(ub, torch.zeros_like(ub)) + mse(vb, torch.zeros_like(vb))

        xt, yt, tt = sample_bc(aux.top_ids)
        loss_bc_top = torch.tensor(0.0, device=device)
        if xt.numel() > 0:
            ru, rv = pinn.bc_top_disp(xt, yt, tt)
            loss_bc_top = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))

        xi, yi, ti = sample_ic_zero()
        ui0, vi0, uvt0, vvt0 = pinn.ic_zero(xi, yi, ti)
        loss_ic = (mse(ui0, torch.zeros_like(ui0)) + mse(vi0, torch.zeros_like(vi0)) +
                   mse(uvt0, torch.zeros_like(uvt0)) + mse(vvt0, torch.zeros_like(vvt0)))

        loss_vib = pinn.bottom_vibration_loss(time_steps=cfg.vib_steps)

        loss = cfg.w_pde * loss_pde + cfg.w_bc * (loss_bc_top) + cfg.w_ic * loss_ic 
        loss.backward()
        opt.step()
        sched.step()

        # ----- outer geometry step (finite difference, fast via adjust_geometry) -----
        if (ep % cfg.geom_every) == 0:
            base_px, base_py, base_x = design.px, design.py, design.xoff
            fd = cfg.geom_fd_eps

            def grad_1d(val, plus_eval, minus_eval):
                step = max(1e-6, abs(val) * fd)
                Jp = plus_eval(step)
                Jm = minus_eval(step)
                return (Jp - Jm) / (2.0 * step)

            # dJ/dpx
            g_px = grad_1d(
                base_px,
                lambda h: evaluate_objective(base_px + h, base_py, base_x),
                lambda h: evaluate_objective(max(1e-9, base_px - h), base_py, base_x),
            )
            # dJ/dpy
            g_py = grad_1d(
                base_py,
                lambda h: evaluate_objective(base_px, base_py + h, base_x),
                lambda h: evaluate_objective(base_px, max(1e-9, base_py - h), base_x),
            )
            # dJ/dxoff
            g_x = grad_1d(
                base_x,
                lambda h: evaluate_objective(base_px, base_py, base_x + h),
                lambda h: evaluate_objective(base_px, base_py, max(0.0, base_x - h)),
            )

            # SGD step
            design.px = max(1e-6, base_px - cfg.geom_lr * g_px)
            design.py = max(1e-6, base_py - cfg.geom_lr * g_py)
            design.xoff = max(0.0,  base_x  - cfg.geom_lr * g_x)

            # commit to Aux once per outer step
            t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            aux.adjust_geometry((design.px, design.py), design.xoff, t_new)
            if hasattr(pinn, "set_mesh"):
                pinn.set_mesh(aux)

        # ----- logs + snapshots -----
        if ep % 10 == 0 or ep == 1:
            t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            print(f"[Ep {ep:05d}] loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"bcT={loss_bc_top.item():.3e} | ic={loss_ic.item():.3e} | vib={loss_vib.item():.3e} | "
                  f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_now:.4f}")

        # (lightweight snapshot)
        outdir = Path("outputs/data"); outdir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            times = np.linspace(0, cfg.T, 21)
            verts = aux.verts_torch.to(device)
            x, y = verts[:, 0], verts[:, 1]
            for k, tk in enumerate(times):
                t_k = torch.full_like(x, fill_value=float(tk))
                u, v = pinn.forward(x, y, t_k)
                disp = torch.stack([u, v], dim=1).cpu().numpy()
                np.savez_compressed(outdir / f"{cfg.name}_{k:04d}.npz", t=float(tk), disp=disp)

    return pinn


# ----------------- Example usage -----------------
if __name__ == "__main__":
    # build your Aux once (connectivity fixed); use options as needed
    aux = Aux(grid_size=(30, 20), pitch=(1.0, 1.2), x_offset=0.2, thickness=0.15, add_diagonals=True)
    pinn = train_pinn(aux, mat=Material(E=20.0, nu=0.3, rho=1.0, plane_stress=True), cfg=TrainCfg())
