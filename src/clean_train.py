import os
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple
import argparse

import numpy as np
import torch
import torch.nn as nn

from utils.smooth_auxetic import *  # provides Aux
from models.fourier_pinn import *    # provides ElastodynamicsPINN
from utils.stateclass import TrainCfg, DesignState, Material
from utils.utils import load_config, set_seed, thickness_from_constraint
import wandb

# ============================ Utilities ==============================

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
    def snapshot_field(self, name: str, pinn: "ElastodynamicsPINN", aux: "Aux", T: float, steps: int = 100, device: Optional[torch.device] = None):
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
    set_seed(1205)
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

    # -------- geometry objective (for FD) --------
    def evaluate_objective(px, py, xoff) -> float:
        # cache current
        px_c, py_c = aux.pitch
        x_c, t_c = aux.x_offset, aux.thickness
        t_new = thickness_from_constraint(design.C, px, py, xoff)

        # set temp geometry
        aux.adjust_geometry((px, py), xoff, t_new)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)

        # what needs autograd inside: PDE / IC / BC need it, vib does not
        use_field_terms = (cfg.geom_use_pde or cfg.geom_use_bc or cfg.geom_use_ic)
        use_vib_term = cfg.geom_use_vib

        cm = torch.enable_grad() if use_field_terms else torch.no_grad()
        with cm:
            J = 0.0

            # PDE / BC / IC terms
            if use_field_terms:
                # PDE
                x_pde, y_pde, t_pde = sample_pde_batch()
                x_pde = x_pde.requires_grad_()
                y_pde = y_pde.requires_grad_()
                t_pde = t_pde.requires_grad_()

                if cfg.geom_use_pde:
                    rx, ry = pinn.pde_residual(x_pde, y_pde, t_pde)
                    pde = mse(rx, torch.zeros_like(rx)) + mse(ry, torch.zeros_like(ry))
                else:
                    pde = torch.tensor(0.0, device=device)

                # BCs
                xt, yt, tt = sample_bc(aux.top_ids)
                if cfg.geom_use_bc and xt.numel() > 0:
                    ru, rv = pinn.bc_top_disp(xt, yt, tt)
                    bc_top = mse(ru, torch.zeros_like(ru)) + mse(rv, torch.zeros_like(rv))
                else:
                    bc_top = torch.tensor(0.0, device=device)

                if cfg.geom_use_bc:
                    bc_bottom = pinn.bc_bottom()
                else:
                    bc_bottom = torch.tensor(0.0, device=device)

                # IC
                xi, yi, ti = sample_ic_zero()
                if cfg.geom_use_ic:
                    ui0, vi0, uvt0, vvt0 = pinn.ic_zero(xi, yi, ti)
                    ic = (mse(ui0, torch.zeros_like(ui0)) + mse(vi0, torch.zeros_like(vi0))
                          + mse(uvt0, torch.zeros_like(uvt0)) + mse(vvt0, torch.zeros_like(vvt0)))
                else:
                    ic = torch.tensor(0.0, device=device)

                J_field = (cfg.w_pde * pde
                           + cfg.w_bc_top * bc_top
                           + cfg.w_bc_bottom * bc_bottom
                           + cfg.w_ic * ic)
                J += float(J_field)

            # vibration term
            if use_vib_term:
                vib = pinn.bottom_vibration_loss(time_steps=cfg.vib_steps)
                J += float(cfg.w_vib * vib)

        # restore geometry
        aux.adjust_geometry((px_c, py_c), x_c, t_c)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)
        return J

    # ---- logging helper ----
    def log(ep_global: int, payload: Dict):
        t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        base = {"geom/px": design.px, "geom/py": design.py, "geom/xoff": design.xoff, "geom/t": t_now}
        base.update(payload)
        wb.log(base, step=ep_global)

    # -------- training (simulation) step --------
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

        # vibration metric
        loss_vib = pinn.bottom_vibration_loss(time_steps=cfg.vib_steps)

        # total simulation loss
        loss = (cfg.w_pde * loss_pde
                + cfg.w_bc_top * loss_bc_top
                + cfg.w_bc_bottom * loss_bc_bottom
                + cfg.w_ic * loss_ic)
        if cfg.sim_use_vib_loss:
            loss = loss + cfg.w_vib * loss_vib

        loss.backward()
        opt.step()
        sched.step()

        if ep_global % cfg.print_every == 0:
            t_now = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
            print(
                f"[SIM {ep_global:05d}] loss={loss.item():.3e} | pde={loss_pde.item():.3e} | "
                f"bcT={loss_bc_top.item():.3e} | bcB={loss_bc_bottom.item():.3e} | "
                f"ic={loss_ic.item():.3e} | vib={loss_vib.item():.3e} | "
                f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_now:.4f}"
            )

        log(ep_global, {
            "train/total_loss": float(loss.item()),
            "train/pde_loss": float(loss_pde.item()),
            "train/bc_top_loss": float(loss_bc_top.item()),
            "train/bc_bottom_loss": float(loss_bc_bottom.item()),
            "train/ic_loss": float(loss_ic.item()),
            "geom/bottom_vibration": float(loss_vib.item()),
            "train/lr": float(opt.param_groups[0]["lr"]),
        })

    # -------- geometry FD step --------
    def geom_step(step_idx: int, ep_global: int):
        base_px, base_py, base_x = design.px, design.py, design.xoff
        fd = cfg.geom_fd_eps

        def grad_1d(val, plus_eval, minus_eval):
            step = max(1e-6, abs(val) * fd)
            Jp = plus_eval(step)
            Jm = minus_eval(step)
            return (Jp - Jm) / (2.0 * step)

        g_px = grad_1d(
            base_px,
            lambda h: evaluate_objective(base_px + h, base_py, base_x),
            lambda h: evaluate_objective(max(1e-9, base_px - h), base_py, base_x),
        )
        g_py = grad_1d(
            base_py,
            lambda h: evaluate_objective(base_px, base_py + h, base_x),
            lambda h: evaluate_objective(base_px, max(1e-9, base_py - h), base_x),
        )
        g_x = grad_1d(
            base_x,
            lambda h: evaluate_objective(base_px, base_py, base_x + h),
            lambda h: evaluate_objective(base_px, base_py, max(0.0, base_x - h)),
        )

        d_px = -cfg.geom_lr * g_px
        d_py = -cfg.geom_lr * g_py
        d_x  = -cfg.geom_lr * g_x

        if cfg.geom_clip and cfg.geom_clip > 0:
            d_px = float(torch.clamp(torch.tensor(d_px), -cfg.geom_clip, cfg.geom_clip))
            d_py = float(torch.clamp(torch.tensor(d_py), -cfg.geom_clip, cfg.geom_clip))
            d_x  = float(torch.clamp(torch.tensor(d_x), -cfg.geom_clip, cfg.geom_clip))

        design.px   = max(1e-6, base_px + d_px)
        design.py   = max(1e-6, base_py + d_py)
        design.xoff = max(0.0,  base_x + d_x)

        # constraint & bounds
        t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
        design.xoff = min(design.xoff, (design.px - t_new) / 2.0)

        aux.adjust_geometry((design.px, design.py), design.xoff, t_new)
        if hasattr(pinn, "set_mesh"):
            pinn.set_mesh(aux)

        if ep_global % cfg.print_every == 0:
            print(
                f"[GEOM {ep_global:05d} | k={step_idx+1}] "
                f"g=({g_px:+.2e},{g_py:+.2e},{g_x:+.2e})  "
                f"Δ=({d_px:+.2e},{d_py:+.2e},{d_x:+.2e})  "
                f"→ px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_new:.4f}"
            )

        J_now = float(evaluate_objective(design.px, design.py, design.xoff))
        if cfg.geom_use_pde or cfg.geom_use_bc or cfg.geom_use_ic:
            key = "metric/combined_objective"
        elif cfg.geom_use_vib:
            key = "metric/bottom_vibration"
        else:
            key = "metric/geom_objective"
        log(ep_global, {key: J_now})

    # ---------------- main loop ----------------
    ep_global = 0
    mode = getattr(cfg, "train_mode", "alternating")

    if mode == "alternating":
        # simulation then geometry each cycle
        for cycle in range(cfg.num_cycles):
            for _ in range(cfg.sim_epochs_per_cycle):
                ep_global += 1
                sim_epoch(ep_global)
            for k in range(cfg.geom_steps_per_cycle):
                ep_global += 1
                geom_step(k, ep_global)

            io.snapshot_field(cfg.name, pinn, aux, cfg.T, steps=100, device=device)
            io.save_state(pinn, design, aux, tag="latest")

    elif mode == "simulation":
        # simulation only
        total_steps = cfg.num_cycles * cfg.sim_epochs_per_cycle
        for s in range(total_steps):
            ep_global += 1
            sim_epoch(ep_global)

            if (s + 1) % (cfg.print_every * 10) == 0:
                io.snapshot_field(cfg.name, pinn, aux, cfg.T, steps=100, device=device)
                io.save_state(pinn, design, aux, tag="latest")

    elif mode == "geometry":
        # geometry only
        total_steps = cfg.num_cycles * cfg.geom_steps_per_cycle
        for s in range(total_steps):
            ep_global += 1
            geom_step(s % cfg.geom_steps_per_cycle, ep_global)

            if (s + 1) % (cfg.print_every * 10) == 0:
                io.snapshot_field(cfg.name, pinn, aux, cfg.T, steps=100, device=device)
                io.save_state(pinn, design, aux, tag="latest")

    else:
        raise ValueError(f"Unknown train_mode '{mode}'")


# ============================ Main Run ============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_json = load_config(args.config)

    # ----- geometry -----
    C = cfg_json["geometry"]["C"]
    grid_size = tuple(cfg_json["geometry"]["grid_size"])
    pitch = tuple(cfg_json["geometry"]["pitch"])
    x_offset = cfg_json["geometry"]["x_offset"]

    thickness = thickness_from_constraint(C, pitch[0], pitch[1], x_offset)

    aux = Aux(
        grid_size=grid_size,
        pitch=pitch,
        x_offset=x_offset,
        thickness=thickness,
        add_diagonals=cfg_json["geometry"]["add_diagonals"],
    )

    # ----- material -----
    z_width = cfg_json["material"]["z_width"]
    rho_scaled = cfg_json["material"]["rho"] * z_width

    mat = Material(
        E=cfg_json["material"]["E"],
        nu=cfg_json["material"]["nu"],
        rho=rho_scaled,
        plane_stress=cfg_json["material"]["plane_stress"]
    )

    # ----- training config -----
    train_cfg = TrainCfg(
        name=cfg_json["training"]["name"],
        T=cfg_json["training"]["T"],
        f_top=cfg_json["training"]["f_top"],
        V0=cfg_json["training"]["V0"],
        bottom_mode=cfg_json["training"]["bottom_mode"],
        m_bottom=cfg_json["training"]["m_bottom"],
        payload_P0=cfg_json["training"]["payload_P0"],

        w_pde=cfg_json["training"]["w_pde"],
        w_bc_top=cfg_json["training"]["w_bc_top"],
        w_bc_bottom=cfg_json["training"]["w_bc_bottom"],
        w_ic=cfg_json["training"]["w_ic"],
        w_vib=cfg_json["training"]["w_vib"],
        lr=cfg_json["training"]["lr"],

        train_mode=cfg_json["training"]["train_mode"],  # "geometry" | "simulation" | "alternating"
        geom_use_pde=cfg_json["training"]["geom_use_pde"],
        geom_use_bc=cfg_json["training"]["geom_use_bc"],      # both top + bottom BC
        geom_use_ic=cfg_json["training"]["geom_use_ic"],
        geom_use_vib=cfg_json["training"]["geom_use_vib"]  ,    # default: old behavior (vib-only geom objective)
    
        sim_use_vib_loss=cfg_json["training"]["sim_use_vib_loss"],  # default: old alternating behavior

        vib_steps=cfg_json["training"]["vib_steps"],
        num_cycles=cfg_json["training"]["num_cycles"],
        sim_epochs_per_cycle=cfg_json["training"]["sim_epochs_per_cycle"],
        geom_steps_per_cycle=cfg_json["training"]["geom_steps_per_cycle"],
        geom_lr=cfg_json["training"]["geom_lr"],
        geom_fd_eps=cfg_json["training"]["geom_fd_eps"],
        geom_clip=cfg_json["training"]["geom_clip"],
        print_every=cfg_json["training"]["print_every"],
        project=cfg_json["training"]["project"],
        tags=tuple(cfg_json["training"]["tags"]),
        notes=cfg_json["training"]["notes"],
    )

    # run
    train_pinn(aux, mat=mat, cfg=train_cfg)


if __name__ == "__main__":
    main()
