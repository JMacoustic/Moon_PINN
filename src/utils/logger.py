import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional
from utils.stateclass import TrainCfg, DesignState, Material
from utils.CST_torch_auxetic import Aux
from models.fourier_galerkin.pinn import *
from models.fourier_galerkin.multifreq_pinn import *


import numpy as np
import torch
import torch.nn as nn
import wandb

from utils.utils import load_config, set_seed, thickness_from_constraint_torch, thickness_from_constraint

# def load_checkpoint(run_name: str, device: str = "cpu"):
#     ckpt_dir = Path("outputs/checkpoints") / run_name
#     cfg_path = ckpt_dir / "config.json"
#     state_path = ckpt_dir / "latest.pt"

#     with open(cfg_path, "r", encoding="utf-8") as f:
#         cfg = json.load(f)

#     train_cfg = TrainCfg(**cfg["train_cfg"])
#     mat_cfg = cfg["material"]
#     material = Material(**mat_cfg)

#     design_cfg = cfg["design_init"]
#     design = DesignState(**design_cfg)

#     mesh_cfg = cfg["mesh"]
#     aux = Aux(
#         grid_size=tuple(mesh_cfg["grid_size"]),
#         pitch=tuple(mesh_cfg["pitch"]),
#         x_offset=mesh_cfg["x_offset"],
#         C_constraint=mesh_cfg["C"],
#         add_diagonals=mesh_cfg["add_diagonals"],
#     ).to(device)

#     model = MultiFourierNodeModel(
#         train_cfg=train_cfg,
#         mesh=aux,
#         device=torch.device(device),
#     ).to(device)
#     model.set_mesh(aux)

#     state = torch.load(state_path, map_location=device)
#     model.load_state_dict(state["model"])
#     aux.load_state_dict(state["mesh"])

#     if "design" in state:
#         for k, v in state["design"].items():
#             setattr(design, k, v)

#     if "material" in state:
#         for k, v in state["material"].items():
#             setattr(material, k, v)

#     return model, design, aux, material

# def snapshot_field(
#     model: nn.Module,
#     mesh,
#     T: float,
#     steps: int = 100,
#     device: torch.device | None = None,
# ):
#     """
#     Evaluate model over [0, T] and return verts_stack & tris as NumPy.

#     Returns
#     -------
#     verts_stack : (steps, N_verts, D)
#     tris        : (N_elems, 3)
#     times       : (steps,)
#     """
#     if device is None:
#         device = next(model.parameters()).device
#     model = model.to(device)
#     mesh = mesh.to(device)

#     times = torch.linspace(0.0, T, steps=steps, device=device)

#     model.eval()
#     with torch.no_grad():
#         disp, _, _ = model.eval_with_derivs(times)  # (steps, N_verts, 2)

#     verts0 = mesh.verts_torch.to(device).detach()      # (N_verts, D)
#     verts_stack = verts0[None, ...] + disp             # (steps, N_verts, D)

#     tris = mesh.tris_torch.detach().cpu().numpy()      # (N_elems, 3)

#     return (
#         verts_stack.detach().cpu().numpy(),  # (T, N, D)
#         tris,                                # (Ne, 3)
#         times.detach().cpu().numpy(),        # (T,)
#     )

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

    def _state_path(self, tag: str) -> Path:
        return self.ckpt_dir / f"{tag}.pt"

    def save_config(self, train_cfg: TrainCfg, material: Material, mesh: Aux, design_init: dict):
        mesh_cfg = {
            "grid_size": mesh.grid_size,
            "pitch": [float(mesh.pitch[0]), float(mesh.pitch[1])],
            "x_offset": float(mesh.x_offset.item() if torch.is_tensor(mesh.x_offset) else mesh.x_offset),
            "C": float(design_init["C"]),
            "add_diagonals": bool(mesh.add_diagonals),
        }
        cfg = {
            "train_cfg": asdict(train_cfg),
            "material": asdict(material),
            "design_init": design_init,
            "mesh": mesh_cfg,
        }
        with open(self.ckpt_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    def save_state(self, model: torch.nn.Module, mesh: Aux,
                   design: DesignState, material: Material, tag: str = "latest") -> None:
        state = {
            "model": model.state_dict(),
            "mesh": mesh.state_dict(),
            "design": asdict(design),
            "material": asdict(material),
        }
        torch.save(state, self._state_path(tag))

    def load_sequence(self, name: str = "latest") -> dict:
        """
        Load a previously saved sequence <name>_*.npz from data_dir.

        Returns:
            {
              "t": (N_steps,) array or None,
              "disp": (N_steps, N_verts, 2) array or None,
            }
        """
        files = sorted(self.data_dir.glob(f"{name}_*.npz"))
        if not files:
            return {"t": None, "disp": None}

        t_list = []
        disp_list = []
        for fpath in files:
            data = np.load(fpath)
            t_list.append(float(data["t"]))
            disp_list.append(data["disp"])

        t_arr = np.asarray(t_list, dtype=float)
        disp_arr = np.stack(disp_list, axis=0)
        return {"t": t_arr, "disp": disp_arr}




    def load_sequence(self, name: str = "latest") -> dict:
        """
        Load a previously saved sequence <name>_*.npz from data_dir.

        Returns:
            {
              "t": (N_steps,) array or None,
              "disp": (N_steps, N_verts, 2) array or None,
            }
        """
        files = sorted(self.data_dir.glob(f"{name}_*.npz"))
        if not files:
            return {"t": None, "disp": None}

        t_list = []
        disp_list = []
        for fpath in files:
            data = np.load(fpath)
            t_list.append(float(data["t"]))
            disp_list.append(data["disp"])

        t_arr = np.asarray(t_list, dtype=float)
        disp_arr = np.stack(disp_list, axis=0)
        return {"t": t_arr, "disp": disp_arr}


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


