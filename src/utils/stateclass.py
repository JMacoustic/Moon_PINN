from dataclasses import dataclass, asdict
import torch
from typing import Dict, Optional, Tuple

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
    z_width: float = 0.1


@dataclass
class TrainCfg:
    name: str = "Default_name"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # time & driving
    T: float = 10.0
    f_top: float = 0.5
    V0: float = 0.05

    # bottom boundary mode
    bottom_mode: str = "payload"   # "mass" | "payload"
    m_bottom: float = 1.0
    payload_P0: float = -0.1

    fourier_K: int = 3 # number of fourier features

    # train loop settings 
    num_cycles: int = 20
    sim_epochs_per_cycle: int = 50
    geom_steps_per_cycle: int = 5
    
    train_mode: str = "alternating"   # "geometry" | "simulation" | "alternating"
    geom_use_pde: bool = False
    geom_use_bc: bool = False      # both top + bottom BC
    geom_use_ic: bool = False
    geom_use_vib: bool = True      # default: old behavior (vib-only geom objective)
    
    sim_use_vib_loss: bool = False  # default: old alternating behavior

    # optimization
    lr: float = 1e-3
    eta_min: float = 1e-6
    pde_batch: int = 8192
    bc_batch: int = 2048
    ic_batch: int = 2048

    # loss weights & metrics
    w_pde: float = 1.0
    w_bc_top: float = 10.0,
    w_bc_bottom: float = 1.0
    w_ic: float = 2.0
    w_vib: float = 1.0
    time_steps: int = 25

    # geometry FD & step control
    geom_lr: float = 1000.0
    geom_fd_eps: float = 1e-3
    geom_clip: float = 5e-3

    # logging
    print_every: int = 10
    project: Optional[str] = None   # None -> no-op logger
    tags: Tuple[str, ...] = ("alt-train",)
    notes: str = ""