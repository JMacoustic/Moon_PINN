import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional
from utils.stateclass import TrainCfg, DesignState, Material
from utils.CST_auxetic import Aux
from models.fourier_galerkin.pinn import *
from models.fourier_galerkin.multifreq_pinn import *


import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.utils import load_config, set_seed, thickness_from_constraint_torch, thickness_from_constraint

def _to_serializable(obj):
    # torch tensors / nn.Parameter
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        else:
            return obj.detach().cpu().tolist()

    # numpy scalar
    if isinstance(obj, np.generic):
        return obj.item()

    # numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # dataclasses / custom objects that have __dict__
    if hasattr(obj, "__dict__"):
        return {k: _to_serializable(v) for k, v in obj.__dict__.items()}

    # dict
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]

    # everything else: leave as-is (must already be JSON-serializable)
    return obj


class Checkpointer:
    """Saves under outputs/checkpoints/<run_name> and outputs/data/<run_name>."""
    def __init__(self, run_name: str):
        self.ckpt_dir = Path("outputs/checkpoints") / run_name
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("outputs/data") / run_name
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, train_cfg, mat, mesh, design_init: Dict):
        cfg = {
            "train": asdict(train_cfg),
            "material": asdict(mat),
            "design_init": design_init,
            "mesh": {
                "grid_size": getattr(mesh, "grid_size", None),
                "pitch": tuple(getattr(mesh, "pitch", (None, None))),
                "x_offset": getattr(mesh, "x_offset", None),
                "thickness": getattr(mesh, "thickness", None),
                "add_diagonals": getattr(mesh, "add_diagonals", None),
            },
        }
        cfg = _to_serializable(cfg)
        with open(self.ckpt_dir / "train_config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_state(self, model: nn.Module, design, mesh, tag: str = "latest"):
        # weights
        torch.save(model.state_dict(), self.ckpt_dir / f"{tag}_weights.pt")

        # geometry
        geom = {
            "px": float(design.px),
            "py": float(design.py),
            "xoff": float(design.xoff),
            "t": float(thickness_from_constraint(design.C, design.px, design.py, design.xoff)),
            "C": float(design.C),
        }
        with open(self.ckpt_dir / f"{tag}_geometry.json", "w", encoding="utf-8") as f:
            json.dump(geom, f, indent=2)

        # mesh (verts + tris)
        np.savez_compressed(
            self.ckpt_dir / f"{tag}_mesh.npz",
            verts=mesh.verts_torch.detach().cpu().numpy(),
            tris=mesh.tris_torch.detach().cpu().numpy(),
        )

    @torch.no_grad()
    def snapshot_field(
        self,
        name: str,
        model: nn.Module,
        mesh,
        T: float,
        steps: int = 100,
        device: Optional[torch.device] = None,
    ):
        """
        Save a SEQUENCE of npz files:
        <name>_0000.npz
        <name>_0001.npz
        ...
        Each contains:
        t: float
        disp: (N_verts, 2)
        """
        device = device or next(model.parameters()).device
        model = model.to(device)

        # time samples
        times = torch.linspace(0.0, T, steps=steps, device=device)

        # eval full displacement series
        disp, _, _ = model.eval_with_derivs(times)   # (steps, N_verts, 2)

        # save each step individually
        for k in range(steps):
            np.savez_compressed(
                self.data_dir / f"{name}_{k:04d}.npz",
                t=float(times[k].item()),
                disp=disp[k].detach().cpu().numpy(),
            )


class WBLogger:
    """No-op compatible wrapper when project=None."""
    def __init__(self, cfg):
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
        if self.enabled:
            wandb.watch(model, **kw)

    def log(self, payload: Dict, step: Optional[int] = None):
        if self.enabled:
            wandb.log(payload, step=step)

    def set_summary(self, **kw):
        if self.enabled:
            for k, v in kw.items():
                self.run.summary[k] = v



def load_checkpoint(run_name: str, device="cuda"):
    ckpt_dir = Path("outputs/checkpoints") / run_name

    # ---- 1. Load config ----
    with open(ckpt_dir / "train_config.json", "r") as f:
        cfg = json.load(f)

    train_cfg = TrainCfg(**cfg["train"])
    mat_cfg   = cfg["material"]
    mesh_cfg  = cfg["mesh"]
    design_init = cfg["design_init"]

    material = Material(**mat_cfg)

    # ---- 2. Load saved geometry ----
    with open(ckpt_dir / "latest_geometry.json", "r") as f:
        geom = json.load(f)

    design = DesignState(
        C=geom["C"],
        px=geom["px"],
        py=geom["py"],
        xoff=geom["xoff"]
    )

    # compute thickness from constraint
    thickness = thickness_from_constraint(design.C, design.px, design.py, design.xoff)

    # ---- 3. Load saved mesh ----
    mesh_npz = np.load(ckpt_dir / "latest_mesh.npz")
    verts = torch.tensor(mesh_npz["verts"], dtype=torch.float32)
    tris  = torch.tensor(mesh_npz["tris"], dtype=torch.long)

    # reconstruct Aux object
    aux = Aux(
        grid_size=mesh_cfg["grid_size"],
        pitch=(design.px, design.py),
        x_offset=design.xoff,
        thickness=thickness,
        add_diagonals=mesh_cfg["add_diagonals"]
    )
    aux.verts_torch = verts
    aux.tris_torch  = tris

    # ---- 4. Load model weights ----
    model = FourierNodeModel(train_cfg=train_cfg, mesh=aux, device=device).to(device)
    state_dict = torch.load(ckpt_dir / "latest_weights.pt", map_location=device)
    model.load_state_dict(state_dict)

    return model, design, aux, train_cfg, material


def load_checkpoint_multi(run_name: str, device="cuda"):
    ckpt_dir = Path("outputs/checkpoints") / run_name

    # ---- 1. Load config ----
    with open(ckpt_dir / "train_config.json", "r") as f:
        cfg = json.load(f)

    train_cfg = TrainCfg(**cfg["train"])
    mat_cfg   = cfg["material"]
    mesh_cfg  = cfg["mesh"]
    design_init = cfg["design_init"]

    material = Material(**mat_cfg)

    # ---- 2. Load saved geometry ----
    with open(ckpt_dir / "latest_geometry.json", "r") as f:
        geom = json.load(f)

    design = DesignState(
        C=geom["C"],
        px=geom["px"],
        py=geom["py"],
        xoff=geom["xoff"]
    )

    # compute thickness from constraint
    thickness = thickness_from_constraint(design.C, design.px, design.py, design.xoff)

    # ---- 3. Load saved mesh ----
    mesh_npz = np.load(ckpt_dir / "latest_mesh.npz")
    verts = torch.tensor(mesh_npz["verts"], dtype=torch.float32)
    tris  = torch.tensor(mesh_npz["tris"], dtype=torch.long)

    # reconstruct Aux object
    aux = Aux(
        grid_size=mesh_cfg["grid_size"],
        pitch=(design.px, design.py),
        x_offset=design.xoff,
        thickness=thickness,
        add_diagonals=mesh_cfg["add_diagonals"]
    )
    aux.verts_torch = verts
    aux.tris_torch  = tris

    # ---- 4. Load model weights ----
    model = MultiFourierNodeModel(train_cfg=train_cfg, mesh=aux, device=device).to(device)
    state_dict = torch.load(ckpt_dir / "latest_weights.pt", map_location=device)
    model.load_state_dict(state_dict)

    return model, design, aux, train_cfg, material
