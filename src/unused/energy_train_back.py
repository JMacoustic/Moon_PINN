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

from utils.CST_torch_auxetic import Aux
from models.fourier_galerkin.pinn import FourierNodeModel
from models.fourier_galerkin.multifreq_pinn import MultiFourierNodeModel
from models.fourier_galerkin.geom_loss import evaluate_geometry_objective
from models.fourier_galerkin.residuals import *
from utils.stateclass import TrainCfg, DesignState, Material
from utils.utils import load_config, set_seed, thickness_from_constraint_torch, thickness_from_constraint
from utils.logger import WBLogger, Checkpointer, load_checkpoint, load_checkpoint_multi

def geometry_fd_step(
    step_idx: int,
    ep_global: int,
    design,
    mesh,
    model,
    train_cfg,
    material,
    device,
    M: torch.Tensor | None = None,
    Kmat: torch.Tensor | None = None,):

    # Ensure mesh geometry matches its current parameters
    mesh.adjust_geometry()
    if hasattr(model, "set_mesh"):
        model.set_mesh(mesh)

    # Evaluate objective (differentiable)
    J = evaluate_geometry_objective(
        mesh=mesh,
        model=model,
        train_cfg=train_cfg,
        material=material,
        device=device,
        M=M,
        Kmat=Kmat
    )

    # For logging only: sync design floats from mesh
    with torch.no_grad():
        # assuming mesh.pitch is shape (2,) tensor, mesh.x_offset, mesh.thickness are tensors
        px = mesh.pitch[0].item()
        py = mesh.pitch[1].item()
        xoff = mesh.x_offset.item()
        t_now = mesh.thickness.item() if hasattr(mesh, "thickness") else None

        design.px   = px
        design.py   = py
        design.xoff = xoff
        if t_now is not None:
            design.t = t_now  # if you have this field

    if ep_global % train_cfg.print_every == 0:
        if hasattr(design, "t"):
            print(
                f"[GEOM {ep_global:05d} | k={step_idx+1}] "
                f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f} t={design.t:.4f}"
            )
        else:
            print(
                f"[GEOM {ep_global:05d} | k={step_idx+1}] "
                f"px={design.px:.4f} py={design.py:.4f} xoff={design.xoff:.4f}"
            )

    # IMPORTANT: return tensor, not float, so caller can do backward()
    return J


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
    Kmat: torch.Tensor | None = None,):

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

    f_verts = torch.zeros_like(mesh.verts_torch, device=device)
    bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)
    if bottom_ids.numel() > 0:
        P0 = train_cfg.payload_P0
        f_verts[bottom_ids, 1] = P0
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
            component="v",
        )
        L_vib_term = train_cfg.w_vib * L_vib
    else:
        with torch.no_grad():
            L_vib = loss_bottom_vibration(
                model,
                mesh,
                component="v",
            )
        L_vib_term = torch.zeros((), device=device)

    loss = (
        train_cfg.w_pde      * L_energy
        # + train_cfg.w_ic     * L_ic
        # + train_cfg.w_bc_top * L_bc
        + train_cfg.w_bc_top * L_vib_term
    )

    loss.backward(retain_graph=True)
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

    do_sim  = mode in ("simulation", "alternating")
    do_geom = mode in ("geometry", "alternating")

    print("mesh.pitch device:", mesh.pitch.device)
    print("mesh.verts_torch device:", mesh.verts_torch.device)
    print("model device:", next(model.parameters()).device)
    print("bottom_ids device:", mesh.bottom_ids.device)

    # --- optimizers / schedulers ---
    # Simulation optimizer: model parameters only (as before)
    if do_sim:
        sim_optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
        total_sim_steps = max(1, train_cfg.num_cycles * train_cfg.sim_epochs_per_cycle)
        sim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            sim_optimizer,
            T_max=total_sim_steps,
            eta_min=train_cfg.eta_min,
        )
    else:
        sim_optimizer = None
        sim_scheduler = None

    # Geometry optimizer: mesh parameters (design parameters) via autograd
    if do_geom:
        geom_optimizer = torch.optim.Adam(mesh.parameters(), lr=train_cfg.geom_lr)
    else:
        geom_optimizer = None

    global_step = 0

    for cycle in range(1, train_cfg.num_cycles + 1):
        print(f"\n=== CYCLE {cycle}/{train_cfg.num_cycles} ===")

        # Ensure mesh verts are consistent with its parameters at cycle start
        mesh.adjust_geometry()
        if hasattr(model, "set_mesh"):
            model.set_mesh(mesh)

        # -----------------------
        # 1) SIMULATION PHASE
        # -----------------------
        if do_sim:
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
                    optimizer=sim_optimizer,
                    scheduler=sim_scheduler,
                    epoch=global_step,
                    M=M,
                    Kmat=Kmat,
                )

                if logger is not None:
                    logger.log(
                        {
                            "lr": sim_optimizer.param_groups[0]["lr"],
                            "loss/total":   sim_out["loss"].item(),
                            "loss/energy":  sim_out["L_energy"].item(),
                            "loss/ic":      sim_out["L_ic"].item(),
                            "loss/bc_top":  sim_out["L_bc"].item(),
                            "loss/vib":     sim_out["L_vib"].item(),
                            "geom/px":   design.px,
                            "geom/py":   design.py,
                            "geom/xoff": design.xoff,
                            "geom/t":    thickness_from_constraint(
                                design.C, design.px, design.py, design.xoff
                            ),
                        },
                        step=global_step,
                    )

        # -----------------------
        # 2) GEOMETRY PHASE
        # -----------------------
        if do_geom:
            for k in range(train_cfg.geom_steps_per_cycle):
                global_step += 1

                geom_optimizer.zero_grad(set_to_none=True)

                # geometry_fd_step returns a torch scalar J (no FD, pure autograd)
                J = geometry_fd_step(
                    step_idx=k,
                    ep_global=global_step,
                    design=design,
                    mesh=mesh,
                    model=model,
                    train_cfg=train_cfg,
                    material=material,
                    device=device,
                )

                J.backward()
                geom_optimizer.step()

                if logger is not None:
                    logger.log(
                        {
                            "geom/J":   J.item(),
                            "geom/px":  design.px,
                            "geom/py":  design.py,
                            "geom/xoff": design.xoff,
                            "geom/t":   thickness_from_constraint(
                                design.C, design.px, design.py, design.xoff
                            ),
                        },
                        step=global_step,
                    )

        # -----------------------
        # 3) CHECKPOINTING
        # -----------------------
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
    ).to(device)


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

    model = MultiFourierNodeModel(
        train_cfg=train_cfg,
        mesh=aux,
        device=device
    ).to(device)

    model.set_mesh(aux)

    if cfg_json["training"]["use_ckpt"] is True:
        model, design, aux, _, mat = load_checkpoint_multi(cfg_json["training"]["ckpt_name"], device="cuda")
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