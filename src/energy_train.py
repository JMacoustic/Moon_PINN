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
from models.fourier_galerkin.pinn import *
from models.fourier_galerkin.geom_loss import evaluate_geometry_objective
from models.fourier_galerkin.residuals import *
from utils.stateclass import TrainCfg, DesignState, Material
from utils.utils import load_config, set_seed, thickness_from_constraint
from utils.logger import WBLogger, Checkpointer, load_checkpoint

def geometry_fd_step(
    step_idx: int,
    ep_global: int,
    design,
    mesh,
    model,
    train_cfg,
    material,
    device,
):
    """
    One finite-difference geometry update step on (px, py, xoff).
    """

    base_px = design.px
    base_py = design.py
    base_x  = design.xoff
    fd      = train_cfg.geom_fd_eps

    def grad_1d(val, plus_eval, minus_eval):
        step = max(1e-6, abs(val) * fd)
        Jp = plus_eval(step)
        Jm = minus_eval(step)
        return (Jp - Jm) / (2.0 * step)

    # central FD for px
    g_px = grad_1d(
        base_px,
        lambda h: evaluate_geometry_objective(
            base_px + h, base_py, base_x,
            design, mesh, model, train_cfg, material, device,
        ),
        lambda h: evaluate_geometry_objective(
            max(1e-9, base_px - h), base_py, base_x,
            design, mesh, model, train_cfg, material, device,
        ),
    )

    # central FD for py
    g_py = grad_1d(
        base_py,
        lambda h: evaluate_geometry_objective(
            base_px, base_py + h, base_x,
            design, mesh, model, train_cfg, material, device,
        ),
        lambda h: evaluate_geometry_objective(
            base_px, max(1e-9, base_py - h), base_x,
            design, mesh, model, train_cfg, material, device,
        ),
    )

    # central FD for xoff
    g_x = grad_1d(
        base_x,
        lambda h: evaluate_geometry_objective(
            base_px, base_py, base_x + h,
            design, mesh, model, train_cfg, material, device,
        ),
        lambda h: evaluate_geometry_objective(
            base_px, base_py, max(0.0, base_x - h),
            design, mesh, model, train_cfg, material, device,
        ),
    )

    # gradient descent step
    d_px = -train_cfg.geom_lr * g_px
    d_py = -train_cfg.geom_lr * g_py
    d_x  = -train_cfg.geom_lr * g_x

    # optional clipping
    if train_cfg.geom_clip and train_cfg.geom_clip > 0:
        d_px = float(torch.clamp(torch.tensor(d_px), -train_cfg.geom_clip, train_cfg.geom_clip))
        d_py = float(torch.clamp(torch.tensor(d_py), -train_cfg.geom_clip, train_cfg.geom_clip))
        d_x  = float(torch.clamp(torch.tensor(d_x),  -train_cfg.geom_clip, train_cfg.geom_clip))

    # update design variables
    design.px   = max(1e-6, base_px + d_px)
    design.py   = max(1e-6, base_py + d_py)
    design.xoff = max(0.0,  base_x + d_x)

    # enforce constraint & bounds
    t_new = thickness_from_constraint(design.C, design.px, design.py, design.xoff)
    design.xoff = min(design.xoff, (design.px - t_new) / 2.0)

    # apply to mesh & model
    mesh.adjust_geometry((design.px, design.py), design.xoff, t_new)
    if hasattr(model, "set_mesh"):
        model.set_mesh(mesh)

    if ep_global % train_cfg.print_every == 0:
        print(
            f"[GEOM {ep_global:05d} | k={step_idx+1}] "
            f"g=({g_px:+.2e},{g_py:+.2e},{g_x:+.2e})  "
            f"Δ=({d_px:+.2e},{d_py:+.2e},{d_x:+.2e})  "
            f"→ px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={t_new:.4f}"
        )

    # current objective value at updated geometry
    J_now = float(evaluate_geometry_objective(
        design.px, design.py, design.xoff,
        design, mesh, model, train_cfg, material, device,
    ))
    return J_now


def sim_train_step(
    design,
    model,
    mesh,
    train_cfg,
    material,
    device,
    optimizer,
    scheduler=None,
    epoch: int = 0,
    M: torch.Tensor | None = None,
    Kmat: torch.Tensor | None = None,
):
    if M is None or Kmat is None:
        M, Kmat = build_cst_mk(
            verts=mesh.verts_torch,
            tris=mesh.tris_torch,
            E=material.E,
            nu=material.nu,
            rho=material.rho,
            th=material.z_width,
        )
        M    = M.to(device)
        Kmat = Kmat.to(device)
    else:
        M    = M.to(device)
        Kmat = Kmat.to(device)

    f_verts = torch.zeros_like(mesh.verts_torch)
    bottom_ids = mesh.bottom_ids
    nb = bottom_ids.numel()
    if nb > 0:
        P0 = train_cfg.payload_P0
        p_each = P0 / float(nb)
        f_verts[bottom_ids, 1] = p_each
    f_verts = f_verts.to(device)

    u0_verts = torch.zeros_like(mesh.verts_torch).to(device)
    bc_ids   = mesh.top_ids.to(device, dtype=torch.long)
    t_batch  = torch.linspace(0.0, train_cfg.T, steps=train_cfg.time_steps, device=device)
    Cmat     = None

    model.train()
    optimizer.zero_grad()

    L_energy = energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=Cmat)
    L_ic     = loss_initial_condition(model, u0_verts)
    L_bc     = loss_boundary_sine(
        model,
        t_batch,
        bc_ids,
        amp_y=train_cfg.V0,
        phase=0.0,
        offset_y=0.0,
        x_fixed=0.0,
    )

    # --- always compute L_vib for logging, only use it in loss if enabled ---
    if train_cfg.sim_use_vib_loss:
        L_vib = loss_bottom_vibration(
            model,
            mesh,
            T=train_cfg.T,
            time_steps=train_cfg.time_steps,
            component="v",
        )
        L_vib_term = train_cfg.w_vib * L_vib
    else:
        with torch.no_grad():
            L_vib = loss_bottom_vibration(
                model,
                mesh,
                T=train_cfg.T,
                time_steps=train_cfg.time_steps,
                component="v",
            )
        L_vib_term = torch.zeros((), device=device)

    loss = (
        train_cfg.w_pde      * L_energy
        # + train_cfg.w_ic     * L_ic
        # + train_cfg.w_bc_top * L_bc
        + train_cfg.w_bc_top * L_vib_term
    )

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    if (epoch % train_cfg.print_every) == 0 or epoch == 1:
        print(
            f"[SIM {epoch:5d}] "
            f"loss={loss.item():.4e} | "
            f"E={L_energy.item():.4e} | "
            f"IC={L_ic.item():.4e} | "
            f"BC={L_bc.item():.4e} | "
            f"VIB={L_vib.item():.4e}"
        )

    return {
        "loss":     loss.detach(),
        "L_energy": L_energy.detach(),
        "L_ic":     L_ic.detach(),
        "L_bc":     L_bc.detach(),
        "L_vib":    L_vib.detach(),   # always meaningful now
        "M":        M,
        "Kmat":     Kmat,
    }


def train_cst_pinn(
    design,
    model,
    mesh,
    train_cfg,
    material,
    device,
    checkpointer: Checkpointer | None = None,
    logger: WBLogger | None = None,
):
    mode = train_cfg.train_mode.lower()
    assert mode in ("simulation", "geometry", "alternating"), f"Unknown train_mode: {mode}"

    # optimizer / scheduler only needed if we do simulation updates
    if mode in ("simulation", "alternating"):
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
        total_sim_steps = max(1, train_cfg.num_cycles * train_cfg.sim_epochs_per_cycle)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_sim_steps,
            eta_min=train_cfg.eta_min,
        )
    else:
        optimizer = None
        scheduler = None

    global_step = 0

    for cycle in range(1, train_cfg.num_cycles + 1):
        print(f"\n=== CYCLE {cycle}/{train_cfg.num_cycles} ===")

        # --- simulation phase ---
        if mode in ("simulation", "alternating"):
            # build FE matrices for current geometry once per cycle
            M, Kmat = build_cst_mk(
                verts=mesh.verts_torch,
                tris=mesh.tris_torch,
                E=material.E,
                nu=material.nu,
                rho=material.rho,
                th=material.z_width,
            )
            M = M.to(device)
            Kmat = Kmat.to(device)

            for i in range(train_cfg.sim_epochs_per_cycle):
                global_step += 1
                sim_out = sim_train_step(
                    design=design,
                    model=model,
                    mesh=mesh,
                    train_cfg=train_cfg,
                    material=material,
                    device=device,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=global_step,
                    M=M,
                    Kmat=Kmat,
                )

                if logger is not None:
                    logger.log(
                        {
                            "lr": optimizer.param_groups[0]["lr"],
                            "loss/total":   sim_out["loss"].item(),
                            "loss/energy":  sim_out["L_energy"].item(),
                            "loss/ic":      sim_out["L_ic"].item(),
                            "loss/bc_top":  sim_out["L_bc"].item(),
                            "loss/vib":     sim_out["L_vib"].item(),

                            # --- geometry logs ---
                            "geom/px":   design.px,
                            "geom/py":   design.py,
                            "geom/xoff": design.xoff,
                            "geom/t":    thickness_from_constraint(design.C, design.px, design.py, design.xoff),
                        },
                        step=global_step,
                    )

        # --- geometry phase ---
        if mode in ("geometry", "alternating"):
            for k in range(train_cfg.geom_steps_per_cycle):
                global_step += 1
                J_now = geometry_fd_step(
                    step_idx=k,
                    ep_global=global_step,
                    design=design,
                    mesh=mesh,
                    model=model,
                    train_cfg=train_cfg,
                    material=material,
                    device=device,
                )

                if logger is not None:
                    logger.log(
                        {
                            # "lr": optimizer.param_groups[0]["lr"],
                            # "loss/total":   sim_out["loss"].item(),
                            # "loss/energy":  sim_out["L_energy"].item(),
                            # "loss/ic":      sim_out["L_ic"].item(),
                            # "loss/bc_top":  sim_out["L_bc"].item(),
                            # "loss/vib":     sim_out["L_vib"].item(),

                            # --- geometry logs ---
                            "geom/px":   design.px,
                            "geom/py":   design.py,
                            "geom/xoff": design.xoff,
                            "geom/t":    thickness_from_constraint(design.C, design.px, design.py, design.xoff),
                        },
                        step=global_step,
                    )

        if checkpointer is not None:
            checkpointer.save_state(model, design, mesh, tag="latest")
            checkpointer.snapshot_field(
                name="latest",
                model=model,
                mesh=mesh,
                T=train_cfg.T,
                steps=100,
                device=device,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_json = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- geometry -----
    C = cfg_json["geometry"]["C"]
    grid_size = tuple(cfg_json["geometry"]["grid_size"])
    pitch = tuple(cfg_json["geometry"]["pitch"])
    x_offset = cfg_json["geometry"]["x_offset"]

    thickness = thickness_from_constraint(C, pitch[0], pitch[1], x_offset)

    design = DesignState(
        C=C,
        px=pitch[0],
        py=pitch[1],
        xoff=x_offset,
    )
    design_init = {
        "px": float(design.px),
        "py": float(design.py),
        "xoff": float(design.xoff),
        "C": float(design.C),
    }

    aux = Aux(
        grid_size=grid_size,
        pitch=pitch,
        x_offset=x_offset,
        thickness=thickness,
        add_diagonals=cfg_json["geometry"]["add_diagonals"],
    )

    # ----- material -----
    

    mat = Material(
        E=cfg_json["material"]["E"],
        nu=cfg_json["material"]["nu"],
        rho=cfg_json["material"]["rho"],
        plane_stress=cfg_json["material"]["plane_stress"],
        z_width = cfg_json["material"]["z_width"],
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

        fourier_K = cfg_json["training"]["fourier_K"],

        w_pde=cfg_json["training"]["w_pde"],
        w_bc_top=cfg_json["training"]["w_bc_top"],
        w_bc_bottom=cfg_json["training"]["w_bc_bottom"],
        w_ic=cfg_json["training"]["w_ic"],
        w_vib=cfg_json["training"]["w_vib"],
        lr=cfg_json["training"]["lr"],
        eta_min =cfg_json["training"]["eta_min"],

        train_mode=cfg_json["training"]["train_mode"],  # "geometry" | "simulation" | "alternating"
        geom_use_pde=cfg_json["training"]["geom_use_pde"],
        geom_use_bc=cfg_json["training"]["geom_use_bc"],      # both top + bottom BC
        geom_use_ic=cfg_json["training"]["geom_use_ic"],
        geom_use_vib=cfg_json["training"]["geom_use_vib"]  ,    # default: old behavior (vib-only geom objective)
    
        sim_use_vib_loss=cfg_json["training"]["sim_use_vib_loss"],  # default: old alternating behavior

        time_steps=cfg_json["training"]["time_steps"],
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

    model = FourierNodeModel(
        train_cfg=train_cfg,
        mesh=aux,
        device=device
    ).to(device)

    if cfg_json["training"]["use_ckpt"] is True:
        model, design, aux, _, mat = load_checkpoint(cfg_json["training"]["ckpt_name"], device="cuda")
        print("successfully loaded checkpoint!")

    ckpt = Checkpointer(train_cfg.name)
    ckpt.save_config(train_cfg, mat, aux, design_init)
    logger = WBLogger(train_cfg)
    logger.watch(model)

    train_cst_pinn(
        design = design,
        model=model,
        mesh=aux,
        train_cfg=train_cfg,
        material=mat,
        device=device,
        checkpointer=ckpt,
        logger=logger
    )

if __name__ == "__main__":
    main()