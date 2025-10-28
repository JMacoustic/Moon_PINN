import os
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.utils.animate import animate  # (unused here but kept if you call it later)
from src.utils.smooth_auxetic import *  # provides Aux
from src.models.simple_pinn import *    # provides ElastodynamicsPINN
import wandb


# =============================== Configs ===============================

@dataclass
class DesignState:
    px: float
    py: float
    xoff: float
    C: float  # constraint constant (px, py, xoff, t satisfy C = (py + 0.5*px + xoff)*t)


@dataclass
class Material:
    E: float = 1.0e1
    nu: float = 0.30
    rho: float = 1.0
    plane_stress: bool = True


@dataclass
class TrainCfg:
    name: str = "Default_name"

    # time & driving
    T: float = 10.0
    f_top: float = 0.5
    V0: float = 0.05

    # bottom boundary mode
    bottom_mode: str = "payload"   # "mass" | "payload"
    m_bottom: float = 1.0
    payload_P0: float = -0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # alternating loop
    num_cycles: int = 20
    sim_epochs_per_cycle: int = 50
    geom_steps_per_cycle: int = 5

    # optimization
    lr: float = 1e-3
    pde_batch: int = 8192
    bc_batch: int = 2048
    ic_batch: int = 2048

    # loss weights & metrics
    w_pde: float = 1.0
    w_bc: float = 10.0
    w_ic: float = 2.0
    w_vib: float = 1.0
    vib_steps: int = 25

    # geometry FD & step control
    geom_lr: float = 1000.0
    geom_fd_eps: float = 1e-3
    geom_clip: float = 5e-3

    # logging
    print_every: int = 10
    project: Optional[str] = None   # None -> no-op logger
    tags: Tuple[str, ...] = ("alt-train",)
    notes: str = ""


# ============================ Utilities ==============================

def thickness_from_constraint(C: float, px: float, py: float, xoff: float) -> float:
    denom = (py + 0.5 * px + xoff)
    return max(1e-9, C / max(1e-9, denom))


class Checkpointer:
    """Handles saving under outputs/checkpoints/<run_name> plus npz snapshots in outputs/data/<run_name>."""
    def __init__(self, run_name: str):
        self.ckpt_dir = Path("outputs/checkpoints") / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("outputs/data") / run_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, train_cfg: TrainCfg, mat: Material, aux: "Aux", design_init: Dict):
        cfg = {
            "train": asdict(train_cfg),
            "material": asdict(mat),
            "design_init": design_init,
            "aux": {
                "grid_size": getattr(aux, "grid_size", None),
                "pitch": tuple(getattr(aux, "pitch", (None, None))),
                "x_offset": getattr(aux, "x_offset", None),
                "thickness": getattr(aux, "thickness", None),
                "add_diagonals": getattr(aux, "add_diagonals", None),
            },
        }
        with open(self.ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_state(self, pinn: nn.Module, design: DesignState, aux: "Aux", tag: str = "latest"):
        torch.save(pinn.state_dict(), self.ckpt_dir / f"{tag}_weights.pt")
        geom = {
            "px": float(design.px),
            "py": float(design.py),
            "xoff": float(design.xoff),
            "t": float(thickness_from_constraint(design.C, design.px, design.py, design.xoff)),
            "C": float(design.C),
        }
        with open(self.ckpt_dir / f"{tag}_geometry.json", "w", encoding="utf-8") as f:
            json.dump(geom, f, indent=2)
        np.savez_compressed(
            self.ckpt_dir / f"{tag}_mesh.npz",
            verts=aux.verts_torch.detach().cpu().numpy(),
            tris=aux.tris_torch.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def snapshot_field(self, name: str, pinn: "ElastodynamicsPINN", aux: "Aux", T: float, steps: int = 21, device: Optional[torch.device] = None):
        device = device or next(pinn.parameters()).device
        times = np.linspace(0, T, steps)
        verts = aux.verts_torch.to(device)
        x, y = verts[:, 0], verts[:, 1]
        for k, tk in enumerate(times):
            t_k = torch.full_like(x, fill_value=float(tk))
            u, v = pinn.forward(x, y, t_k)
            disp = torch.stack([u, v], dim=1).cpu().numpy()
            np.savez_compressed(self.data_dir / f"{name}_{k:04d}.npz", t=float(tk), disp=disp)


class WBLogger:
    """No-op compatible wrapper when project=None."""
    def __init__(self, cfg: TrainCfg):
        self.enabled = cfg.project is not None
        if self.enabled:
            self.run = wandb.init(
                project=cfg.project,
                name=cfg.name,
                config={"train": asdict(cfg)},
                tags=list(cfg.tags) if cfg.tags else None,
                notes=cfg.notes or None,
            )
        else:
            self.run = None

    def watch(self, model: nn.Module, **kw):
        if self.enabled: wandb.watch(model, **kw)

    def log(self, payload: Dict, step: Optional[int] = None):
        if self.enabled: wandb.log(payload, step=step)

    def set_summary(self, **kw):
        if self.enabled:
            for k, v in kw.items():
                self.run.summary[k] = v


# ============================ Training ==============================

def train_pinn(aux: "Aux", mat: Material, cfg: TrainCfg):
    os.makedirs("outputs", exist_ok=True)
    device = torch.device(cfg.device)

    # ---- initialize design from Aux ----
    px0, py0 = aux.pitch
    xoff0 = aux.x_offset
    t0 = aux.thickness
    C0 = (py0 + 0.5 * px0 + xoff0) * t0
    design = DesignState(px=px0, py=py0, xoff=xoff0, C=C0)

    # ---- IO setup ----
    io = Checkpointer(cfg.name)
    io.save_config(cfg, mat, aux, {"px": px0, "py": py0, "xoff": xoff0, "t": t0, "C": C0})
    wb = WBLogger(cfg)

    # ---- build model & optim ----
    pinn = ElastodynamicsPINN(aux, mat, cfg).to(device)
    opt = torch.optim.Adam(pinn.parameters(), lr=cfg.lr)
    total_sim_steps = cfg.num_cycles * cfg.sim_epochs_per_cycle
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_sim_steps, eta_min=1e-6)
    mse = nn.MSELoss()
    wb.watch(pinn, log="all", log_freq=50)

    # -------- samplers --------
    def sample_pde_batch():
        P = aux.sample_interior(cfg.pde_batch).to(device)
        t = cfg.T * torch.rand(P.shape[0], 1, device=device)
        return P[:, 0], P[:, 1], t.view(-1)

    def sample_bc(ids):
        P = aux.sample_on_nodes(ids, cfg.bc_batch).to(device)
        if P.numel() == 0:
            z = torch.empty(0, device=device)
            return z, z, z
        t = cfg.T * torch.rand(P.shape[0], 1, device=device)
        return P[:, 0], P[:, 1], t.view(-1)

    def sample_ic_zero():
        P = aux.sample_interior(cfg.ic_batch).to(device)
        t0 = torch.zeros(P.shape[0], device=device)
        return P[:, 0], P[:, 1], t0

    # -------- objective (temporary geometry) --------
    @torch.no_grad()
    def evaluate_objective(px, py, xoff) -> float:
        px_c, py_c = aux.pitch
        x_c, t_c = aux.x_offset, aux.thickness
        t_new = thickness_from_constraint(design.C, px, py, xoff)

        aux.adjust_geometry((px, py), xoff, t_new)
        if hasattr(pinn, "set_mesh"): pinn.set_mesh(aux)
        J = float(pinn.bottom_vibration_loss(time_steps=cfg.vib_steps))

        aux.adjust_geometry((px_c, py_c), x_c, t_c)
        if hasattr(pinn, "set_mesh"): pinn.set_mesh(aux)
        return J

    # ---- logging helper ----
    def log(ep_global: int, payload: Dict):
        t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        base = {"geom/px": design.px, "geom/py": design.py, "geom/xoff": design.xoff, "geom/t": t_now}
        base.update(payload)
        wb.log(base, step=ep_global)

    # -------- training steps --------
    def sim_epoch(ep_global: int):
        opt.zero_grad(set_to_none=True)

        # PDE residual
        x, y, t = sample_pde_batch()
        rx, ry = pinn.pde_residual(x, y, t)
        loss_pde = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry))

        # BCs
        xt, yt, tt = sample_bc(aux.top_ids)
        loss_bc_top = torch.tensor(0.0, device=device)
        if xt.numel() > 0:
            ru, rv = pinn.bc_top_disp(xt, yt, tt)
            loss_bc_top = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))

        loss_bc_bottom = pinn.bc_bottom()

        # IC (zero)
        xi, yi, ti = sample_ic_zero()
        ui0, vi0, uvt0, vvt0 = pinn.ic_zero(xi, yi, ti)
        loss_ic = (mse(ui0, torch.zeros_like(ui0)) + mse(vi0, torch.zeros_like(vi0))
                   + mse(uvt0, torch.zeros_like(uvt0)) + mse(vvt0, torch.zeros_like(vvt0)))

        # metric (not in loss, but tracked)
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

        log(ep_global, {
            "train/total_loss": float(loss.item()),
            "train/pde_loss": float(loss_pde.item()),
            "train/bc_top_loss": float(loss_bc_top.item()),
            "train/bc_bottom_loss": float(loss_bc_bottom.item()),
            "train/ic_loss": float(loss_ic.item()),
            "geom/bottom_vibration": float(loss_vib.item()),
            "train/lr": float(opt.param_groups[0]["lr"]),
        })

    def geom_step(step_idx: int, ep_global: int):
        base_px, base_py, base_x = design.px, design.py, design.xoff
        fd = cfg.geom_fd_eps

        def grad_1d(val, plus_eval, minus_eval):
            step = max(1e-6, abs(val) * fd)
            Jp = plus_eval(step)
            Jm = minus_eval(step)
            return (Jp - Jm) / (2.0 * step)

        g_px = grad_1d(base_px,
                       lambda h: evaluate_objective(base_px + h, base_py, base_x),
                       lambda h: evaluate_objective(max(1e-9, base_px - h), base_py, base_x))
        g_py = grad_1d(base_py,
                       lambda h: evaluate_objective(base_px, base_py + h, base_x),
                       lambda h: evaluate_objective(base_px, max(1e-9, base_py - h), base_x))
        g_x  = grad_1d(base_x,
                       lambda h: evaluate_objective(base_px, base_py, base_x + h),
                       lambda h: evaluate_objective(base_px, base_py, max(0.0, base_x - h)))

        d_px = -cfg.geom_lr * g_px
        d_py = -cfg.geom_lr * g_py
        d_x  = -cfg.geom_lr * g_x

        if cfg.geom_clip and cfg.geom_clip > 0:
            d_px = float(torch.clamp(torch.tensor(d_px), -cfg.geom_clip, cfg.geom_clip))
            d_py = float(torch.clamp(torch.tensor(d_py), -cfg.geom_clip, cfg.geom_clip))
            d_x  = float(torch.clamp(torch.tensor(d_x),  -cfg.geom_clip, cfg.geom_clip))

        design.px   = max(1e-6, base_px + d_px)
        design.py   = max(1e-6, base_py + d_py)
        design.xoff = max(0.0,  base_x + d_x)

        # update Aux (with constraint t = C/(py + 0.5*px + xoff))
        t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        # enforce 0 <= x_offset <= (px - t)/2
        design.xoff = min(design.xoff, (design.px - t_new) / 2.0)

        aux.adjust_geometry((design.px, design.py), design.xoff, t_new)
        if hasattr(pinn, "set_mesh"): pinn.set_mesh(aux)

        if ep_global % cfg.print_every == 0:
            print(f"[GEOM {ep_global:05d} | k={step_idx+1}] "
                  f"g=({g_px:+.2e},{g_py:+.2e},{g_x:+.2e})  "
                  f"Δ=({d_px:+.2e},{d_py:+.2e},{d_x:+.2e})  "
                  f"→ px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_new:.4f}")

        J_now = float(evaluate_objective(design.px, design.py, design.xoff))
        log(ep_global, {"metric/bottom_vibration": J_now})

    # ---------------- main alternating loop ----------------
    ep_global = 0
    for cycle in range(cfg.num_cycles):
        for _ in range(cfg.sim_epochs_per_cycle):
            ep_global += 1
            sim_epoch(ep_global)
        for k in range(cfg.geom_steps_per_cycle):
            ep_global += 1
            geom_step(k, ep_global)

        io.snapshot_field(cfg.name, pinn, aux, cfg.T, steps=21, device=device)
        io.save_state(pinn, design, aux, tag="latest")

    # ---- finalize ----
    t_final = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
    wb.set_summary(final__px=design.px, final__py=design.py, final__xoff=design.xoff, final__t=t_final)

    io.snapshot_field(cfg.name, pinn, aux, cfg.T, steps=21, device=device)
    io.save_state(pinn, design, aux, tag="final")
    return pinn


# ============================ Example run ============================

if __name__ == "__main__":
    # geometry/constraint
    C = 0.002
    grid_size = (10, 10)
    pitch = (0.1, 0.1)
    x_offset = 0.05
    thickness = thickness_from_constraint(C, pitch[0], pitch[1], x_offset)

    # z-direction width (for rho scaling example)
    z_width = 0.1

    MaterialCfg = Material(
        E=2e9,
        nu=0.35,
        rho=1240.0 * z_width,
        plane_stress=True,
    )

    cfg = TrainCfg(
        name="251028_normalized_zwidth_v1",
        T=10.0,
        f_top=5.0,
        V0=0.001,
        bottom_mode="payload",
        m_bottom=0.5,
        payload_P0=-5.0,

        w_pde=1.0,
        w_bc=2.0,
        w_ic=2.0,
        w_vib=1.0,
        lr=1e-4,

        vib_steps=25,
        num_cycles=1000,
        sim_epochs_per_cycle=100,
        geom_steps_per_cycle=10,
        geom_lr=100.0,
        geom_fd_eps=1e-3,
        geom_clip=2e-4,
        print_every=10,
        project="auxetic-pinn",                 # set None to disable W&B
        tags=("pinn", "geom-opt"),
        notes="Alternating sim/geom with FD grads; logs geometry + losses + vibration.",
    )

    aux = Aux(grid_size=grid_size, pitch=pitch, x_offset=x_offset, thickness=thickness, add_diagonals=True)
    _ = train_pinn(aux, mat=MaterialCfg, cfg=cfg)
