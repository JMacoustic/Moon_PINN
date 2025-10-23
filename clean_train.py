import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from src.utils.animate import animate
from src.utils.smooth_auxetic import *
from src.models.simple_pinn import *
import wandb
import json 

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
    name: str = "251001_testrun_v1"
    # simulation horizon & driving
    T: float = 10.0
    f_top: float = 0.5
    V0: float = 0.05
    m_bottom: float = 1
    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # phase scheduling
    num_cycles: int = 20
    sim_epochs_per_cycle: int = 50
    geom_steps_per_cycle: int = 5
    # PINN opt
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
    # geometry hyperparams
    geom_lr: float = 1000.0
    geom_fd_eps: float = 1e-3
    geom_clip: float = 5e-3
    # I/O cadence
    print_every: int = 10
    snapshot_every: int = 50
    # wandb
    project: str = None      # set to your project name or leave None to use env WANDB_PROJECT
    tags: tuple = ("alt-train",)
    notes: str = ""

    # === SAVE HELPERS (minimal) ===
def _ckpt_dir():
        d = Path("outputs/checkpoints") / cfg.name
        d.mkdir(parents=True, exist_ok=True)
        return d

def _save(design, ep_global: int = None, tag: str | None = None):
        d = _ckpt_dir()
        tag = tag or "latest"  # default to overwriting "latest"

        # 1) weights (single file overwritten)
        torch.save(pinn.state_dict(), d / f"{tag}_weights.pt")

        # 2) geometry (single file overwritten)
        geom = {
            "px": float(design.px),
            "py": float(design.py),
            "xoff": float(design.xoff),
            "t": float(thickness_from_constraint(design.C, design.px, design.py, design.xoff)),
            "C": float(design.C),
        }
        with open(d / f"{tag}_geometry.json", "w", encoding="utf-8") as f:
            json.dump(geom, f, indent=2)

        # 3) mesh (single file overwritten)
        np.savez_compressed(
            d / f"{tag}_mesh.npz",
            verts=aux.verts_torch.detach().cpu().numpy(),
            tris=aux.tris_torch.detach().cpu().numpy(),
        )

def thickness_from_constraint(C: float, px: float, py: float, xoff: float) -> float:
    denom = (py + 0.5 * px + xoff)
    return max(1e-9, C / max(1e-9, denom))

def train_pinn(aux: "Aux", mat=Material(), cfg=TrainCfg()):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device(cfg.device)

    # ---- init design state from aux ----
    px0, py0 = aux.pitch
    xoff0 = aux.x_offset
    t_init = aux.thickness
    C_const = (py0 + 0.5 * px0 + xoff0) * t_init
    design = DesignState(px=px0, py=py0, xoff=xoff0, C=C_const)

    # ---- wandb init ----
    run = wandb.init(
        project=cfg.project,
        name=cfg.name,
        config={
            "train": asdict(cfg),
            "material": asdict(mat),
            "design_init": {"px": px0, "py": py0, "xoff": xoff0, "t": t_init, "C": C_const}},
        tags=list(cfg.tags) if cfg.tags else None,
        notes=cfg.notes or None,
    )

    # ---- build PINN ----
    pinn = ElastodynamicsPINN(aux, mat, cfg).to(device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.num_cycles*cfg.sim_epochs_per_cycle, eta_min=1e-6)
    mse = nn.MSELoss()
    wandb.watch(pinn, log="all", log_freq=50)

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

    # -------- objective under temporary geometry --------
    @torch.no_grad()
    def evaluate_objective(px, py, xoff) -> float:
        px_c, py_c = aux.pitch
        x_c = aux.x_offset
        t_c = aux.thickness

        t_new = thickness_from_constraint(design.C, px, py, xoff)
        aux.adjust_geometry((px, py), xoff, t_new)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)

        J = float(pinn.bottom_vibration_loss(time_steps=cfg.vib_steps))

        aux.adjust_geometry((px_c, py_c), x_c, t_c)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)
        return J

    # ---- logging helper ----
    def log(ep_global: int, payload: Dict):
        # always append current geometry
        t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        base = {
            "geom/px": design.px,
            "geom/py": design.py,
            "geom/xoff": design.xoff,
            "geom/t": t_now,
        }
        base.update(payload)
        wandb.log(base, step=ep_global)

    # -------- helpers --------
    def sim_epoch(ep_global: int):
        opt.zero_grad(set_to_none=True)

        # PDE
        x, y, t = sample_pde_batch()
        rx, ry = pinn.pde_residual(x, y, t)
        loss_pde = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry))

        # BCs
        xt, yt, tt = sample_bc(aux.top_ids)
        loss_bc_top = torch.tensor(0.0, device=device)
        if xt.numel() > 0:
            ru, rv = pinn.bc_top_disp(xt, yt, tt)
            loss_bc_top = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))

        loss_bc_bottom = pinn.bc_bottom_mass()

        # IC
        xi, yi, ti = sample_ic_zero()
        ui0, vi0, uvt0, vvt0 = pinn.ic_zero(xi, yi, ti)
        loss_ic = (mse(ui0, torch.zeros_like(ui0)) + mse(vi0, torch.zeros_like(vi0)) +
                   mse(uvt0, torch.zeros_like(uvt0)) + mse(vvt0, torch.zeros_like(vvt0)))

        # optional metric
        loss_vib = pinn.bottom_vibration_loss(time_steps=cfg.vib_steps)

        loss = cfg.w_pde * loss_pde + cfg.w_bc * (loss_bc_top + loss_bc_bottom) + cfg.w_ic * loss_ic
        loss.backward()
        opt.step()
        sched.step()

        if ep_global % cfg.print_every == 0:
            t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            print(f"[SIM {ep_global:05d}] loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                  f"bcT={loss_bc_top.item():.3e} | bcB={loss_bc_bottom.item():.3e} | "
                  f"ic={loss_ic.item():.3e} | vib={loss_vib.item():.3e} | "
                  f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_now:.4f}")

        # --- W&B log (simulation) ---
        log(ep_global, {
            "train/total_loss": float(loss.item()),
            "train/pde_loss": float(loss_pde.item()),
            "train/bc_top_loss": float(loss_bc_top.item()),
            "train/bc_bottom_loss": float(loss_bc_bottom.item()),
            "train/ic_loss": float(loss_ic.item()),
            "geom/bottom_vibration": float(loss_vib.item()),
            "train/lr": float(opt.param_groups[0]["lr"]),
        })

        if ep_global % cfg.snapshot_every == 0:
            snapshot(ep_global)

    def geom_step(step_idx: int, ep_global: int):
        base_px, base_py, base_x = design.px, design.py, design.xoff
        fd = cfg.geom_fd_eps

        def grad_1d(val, plus_eval, minus_eval):
            step = max(1e-6, abs(val) * fd)
            Jp = plus_eval(step)
            Jm = minus_eval(step)
            return (Jp - Jm) / (2.0 * step), step, Jp, Jm

        # dJ/dpx
        g_px, h_px, Jp_px, Jm_px = grad_1d(
            base_px,
            lambda h: evaluate_objective(base_px + h, base_py, base_x),
            lambda h: evaluate_objective(max(1e-9, base_px - h), base_py, base_x),
        )
        # dJ/dpy
        g_py, h_py, Jp_py, Jm_py = grad_1d(
            base_py,
            lambda h: evaluate_objective(base_px, base_py + h, base_x),
            lambda h: evaluate_objective(base_px, max(1e-9, base_py - h), base_x),
        )
        # dJ/dxoff
        g_x, h_x, Jp_x, Jm_x = grad_1d(
            base_x,
            lambda h: evaluate_objective(base_px, base_py, base_x + h),
            lambda h: evaluate_objective(base_px, base_py, max(0.0, base_x - h)),
        )

        # SGD step on geometry
        d_px = -cfg.geom_lr * g_px
        d_py = -cfg.geom_lr * g_py
        d_x  = -cfg.geom_lr * g_x

        if cfg.geom_clip and cfg.geom_clip > 0:
            d_px = float(torch.clamp(torch.tensor(d_px), -cfg.geom_clip, cfg.geom_clip))
            d_py = float(torch.clamp(torch.tensor(d_py), -cfg.geom_clip, cfg.geom_clip))
            d_x  = float(torch.clamp(torch.tensor(d_x),  -cfg.geom_clip, cfg.geom_clip))

        design.px   = max(1e-6, base_px + d_px)
        design.py   = max(1e-6, base_py + d_py)
        design.xoff = max(0.0,  base_x  + d_x)

        # commit update to Aux
        t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        aux.adjust_geometry((design.px, design.py), design.xoff, t_new)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)

        if ep_global % cfg.print_every == 0:
            print(f"[GEOM {ep_global:05d} | k={step_idx+1}] "
                  f"g(px,py,x)={(g_px):+.2e},{(g_py):+.2e},{(g_x):+.2e}  "
                  f"d={(d_px):+.2e},{(d_py):+.2e},{(d_x):+.2e}  "
                  f"→ px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_new:.4f}")

        # evaluate vibration at new geometry for logging
        J_now = float(evaluate_objective(design.px, design.py, design.xoff))

        # --- W&B log (geometry) ---
        log(ep_global, {
            "metric/bottom_vibration": float(J_now),
        })

    @torch.no_grad()
    def snapshot(ep_global: int):
        outdir = Path("outputs/data"); outdir.mkdir(parents=True, exist_ok=True)
        times = np.linspace(0, cfg.T, 21)
        verts = aux.verts_torch.to(device)
        x, y = verts[:, 0], verts[:, 1]
        for k, tk in enumerate(times):
            t_k = torch.full_like(x, fill_value=float(tk))
            u, v = pinn.forward(x, y, t_k)
            disp = torch.stack([u, v], dim=1).cpu().numpy()
            np.savez_compressed(outdir / f"{cfg.name}_ep{ep_global:05d}_{k:04d}.npz", t=float(tk), disp=disp)


    # ---------------- main alternating loop ----------------
    ep_global = 0
    for cycle in range(cfg.num_cycles):
        for _ in range(cfg.sim_epochs_per_cycle):
            ep_global += 1
            sim_epoch(ep_global)
        for k in range(cfg.geom_steps_per_cycle):
            ep_global += 1
            geom_step(k, ep_global)
        snapshot(ep_global)
        _save(design, tag = "latest")

    # summarize final design
    t_final = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
    run.summary["final/px"] = design.px
    run.summary["final/py"] = design.py
    run.summary["final/xoff"] = design.xoff
    run.summary["final/t"] = t_final

    _save(design, tag = "final")
    return pinn

# ----------------- Example usage -----------------
if __name__ == "__main__":
    aux = Aux(grid_size=(10, 10), pitch=(1.0, 1.2), x_offset=0.2, thickness=0.15, add_diagonals=True)
    cfg = TrainCfg(
        name="251023_testrun_v1",
        num_cycles=1000,
        sim_epochs_per_cycle=50,
        geom_steps_per_cycle=10,
        geom_lr=1000.0,
        geom_fd_eps=1e-3,
        geom_clip=2e-3,
        print_every=10,
        snapshot_every=50,
        project="auxetic-pinn",
        tags=("pinn", "geom-opt"),
        notes="Alternating sim/geom with FD grads; logs geometry + losses + vibration.",
    )
    pinn = train_pinn(aux, mat=Material(E=20.0, nu=0.3, rho=1.0, plane_stress=True), cfg=cfg)
