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
# from models.fourier_galerkin.geom_loss import evaluate_geometry_objective
from models.fourier_galerkin.residuals import *
from utils.stateclass import TrainCfg, DesignState, Material
from utils.utils import *
from utils.logger import WBLogger, Checkpointer

def load_checkpoint(run_name: str, device: str = "cpu"):
    ckpt_dir = Path("outputs/checkpoints") / run_name
    cfg_path = ckpt_dir / "config.json"
    state_path = ckpt_dir / "latest.pt"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    train_cfg = TrainCfg(**cfg["train_cfg"])
    mat_cfg = cfg["material"]
    material = Material(**mat_cfg)

    design_cfg = cfg["design_init"]
    design = DesignState(**design_cfg)

    mesh_cfg = cfg["mesh"]
    aux = Aux(
        grid_size=tuple(mesh_cfg["grid_size"]),
        pitch=tuple(mesh_cfg["pitch"]),
        x_offset=mesh_cfg["x_offset"],
        C_constraint=mesh_cfg["C"],
        add_diagonals=mesh_cfg["add_diagonals"],
    ).to(device)

    model = MultiFourierNodeModel(
        train_cfg=train_cfg,
        mesh=aux,
        device=torch.device(device),
    ).to(device)
    model.set_mesh(aux)

    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state["model"])
    aux.load_state_dict(state["mesh"])

    if "design" in state:
        for k, v in state["design"].items():
            setattr(design, k, v)

    if "material" in state:
        for k, v in state["material"].items():
            setattr(material, k, v)

    return model, design, aux, material, train_cfg



def geometry_step(
    step_idx: int,
    epoch: int,
    design,
    mesh,
    model,
    train_cfg,
    material,
    device,
    optimizer,
    scheduler=None):

    # Keep mesh/model in sync (verts_torch depends on pitch, x_offset, thickness)
    mesh.adjust_geometry()
    if hasattr(model, "set_mesh"):
        model.set_mesh(mesh)

    # FE matrices depend on geometry AND thickness → build inside step so grads flow
    M, Kmat = build_cst_mk(
        verts=mesh.verts_torch,
        tris=mesh.tris_torch,
        E=material.E,
        nu=material.nu,
        rho=material.rho,
        z_width=material.z_width)

    M    = M.to(device)
    Kmat = Kmat.to(device)

    # External nodal forces
    f_verts = torch.zeros_like(mesh.verts_torch, device=device)
    bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)
    if bottom_ids.numel() > 0:
        P0 = train_cfg.payload_P0
        f_verts[bottom_ids, 1] = P0

    # Time grid, IC, BC ids
    t_batch  = torch.linspace(0.0, train_cfg.T, steps=train_cfg.time_steps, device=device)
    u0_verts = torch.zeros_like(mesh.verts_torch, device=device)
    bc_ids   = mesh.top_ids.to(device, dtype=torch.long)

    # Loss terms (geom flags)
    use_field_terms = train_cfg.geom_use_pde
    use_vib_term    = train_cfg.geom_use_vib

    if use_field_terms:
        L_energy = energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None)
    else:
        L_energy = torch.zeros((), device=device)

    if use_vib_term:
        L_vib = loss_bottom_vibration(model, mesh, component="v")
    else:
        L_vib = torch.zeros((), device=device)
    
    L_coll = loss_collision(mesh)

    J = train_cfg.w_pde * L_energy + train_cfg.w_vib * L_vib + train_cfg.w_coll * L_coll

    # Optimizer & scheduler
    optimizer.zero_grad()
    J.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    # Sync design floats from mesh (for logging / constraints)
    with torch.no_grad():
        px   = mesh.pitch[0].item()
        py   = mesh.pitch[1].item()
        xoff = mesh.x_offset.item()
        t_now = mesh.thickness.item()

        design.px   = px
        design.py   = py
        design.xoff = xoff

    # Console logging similar to sim_train_step
    if (epoch % train_cfg.print_every) == 0 or epoch == 1:
        print(
            f"[GEOM {epoch:5d} | k={step_idx+1}] "
            f"loss={J.item():.4e} | "
            f"E={L_energy.item():.4e} | "
            f"VIB={L_vib.item():.4e} | "
            f"px={design.px:.4f} py={design.py:.4f} "
            f"xoff={design.xoff:.4f} t={t_now:.4f}"
        )

    return {
        "loss":     J.detach(),
        "L_energy": L_energy.detach(),
        "L_vib":    L_vib.detach(),
    }



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
            z_width=material.z_width)
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

    if train_cfg.sim_use_vib_loss:
        L_vib = loss_bottom_vibration(model, mesh,component="v")
        L_vib_term = L_vib
    else:
        with torch.no_grad():
            L_vib = loss_bottom_vibration(model, mesh, component="v")
            L_vib_term = torch.zeros((), device=device)

    loss = train_cfg.w_pde * L_energy + train_cfg.w_vib * L_vib_term
    loss.backward(retain_graph=True)
    optimizer.step()
    scheduler.step()

    if (epoch % train_cfg.print_every) == 0 or epoch == 1:
        print(
            f"[SIM {epoch:5d}] "
            f"loss={loss.item():.4e} | "
            f"E={L_energy.item():.4e} | "
            f"VIB={L_vib.item():.4e}"
        )

    return {
        "loss":     loss.detach(),
        "L_energy": L_energy.detach(),
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
    logger: WBLogger | None = None):

    mode = train_cfg.train_mode.lower()
    assert mode in ("simulation", "geometry", "alternating"), f"Unknown train_mode: {mode}"

    do_sim  = mode in ("simulation", "alternating")
    do_geom = mode in ("geometry", "alternating")

    # --- SIM optimizer/scheduler ---
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

    # --- GEOM optimizer/scheduler ---
    if do_geom:
        geom_optimizer = torch.optim.Adam(mesh.parameters(), lr=train_cfg.geom_lr)
        total_geom_steps = max(1, train_cfg.num_cycles * train_cfg.geom_steps_per_cycle)
        geom_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            geom_optimizer,
            T_max=total_geom_steps,
            eta_min=train_cfg.eta_min,
        )
    else:
        geom_optimizer = None
        geom_scheduler = None

    global_step = 0

    for cycle in range(1, train_cfg.num_cycles + 1):
        print(f"\n=== CYCLE {cycle}/{train_cfg.num_cycles} ===")

        mesh.adjust_geometry()
        if hasattr(model, "set_mesh"):
            model.set_mesh(mesh)

        # -----------------------
        # 1) SIMULATION PHASE
        # -----------------------
        if do_sim:
            M, Kmat = build_cst_mk(
                verts=mesh.verts_torch,
                tris=mesh.tris_torch,
                E=material.E,
                nu=material.nu,
                rho=material.rho,
                z_width=material.z_width,
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
                    Kmat=Kmat)

                if logger is not None:
                    logger.log(
                        {
                            "lr/sim": sim_optimizer.param_groups[0]["lr"],
                            "loss/total":   sim_out["loss"].item(),
                            "loss/energy":  sim_out["L_energy"].item(),
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

                geom_out = geometry_step(
                    step_idx=k,
                    epoch=global_step,
                    design=design,
                    mesh=mesh,
                    model=model,
                    train_cfg=train_cfg,
                    material=material,
                    device=device,
                    optimizer=geom_optimizer,
                    scheduler=geom_scheduler,
                )

                if logger is not None:
                    logger.log(
                        {
                            "lr/geom":  geom_optimizer.param_groups[0]["lr"],
                            "geom/J":   geom_out["loss"].item(),
                            "geom/E":   geom_out["L_energy"].item(),
                            "geom/VIB": geom_out["L_vib"].item(),
                            "geom/px":  design.px,
                            "geom/py":  design.py,
                            "geom/xoff": design.xoff,
                            "geom/t":   thickness_from_constraint(
                                design.C, design.px, design.py, design.xoff),
                        },
                        step=global_step,
                    )

        # 3) CHECKPOINTING
        if checkpointer is not None:
            checkpointer.save_state(model, mesh, design, material, tag="latest")

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
        C_constraint=C,
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
        w_vib=cfg_json["training"]["w_vib"],
        w_coll=cfg_json["training"]["w_coll"],
        lr=cfg_json["training"]["lr"],
        eta_min =cfg_json["training"]["eta_min"],
        train_mode=cfg_json["training"]["train_mode"], 
        geom_use_pde=cfg_json["training"]["geom_use_pde"],
        geom_use_vib=cfg_json["training"]["geom_use_vib"],
        sim_use_vib_loss=cfg_json["training"]["sim_use_vib_loss"], 
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
        model, design, aux, mat, _ = load_checkpoint(cfg_json["training"]["ckpt_name"], device="cuda")
        print("successfully loaded checkpoint!")

    ckpt = Checkpointer(train_cfg.name)
    ckpt.save_config(train_cfg, mat, aux, design_init)
    logger = WBLogger(train_cfg)
    logger.watch(model)

    train_cst_pinn(
        design=design,
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