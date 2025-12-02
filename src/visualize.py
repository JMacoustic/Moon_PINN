# visualize_vibration.py
import argparse
import json
from pathlib import Path
import torch

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import imageio.v2 as imageio
from utils.utils import *
from utils.stateclass import *
from models.fourier_galerkin.multifreq_pinn import MultiFourierNodeModel
from utils.CST_torch_auxetic import Aux

# ---------- mesh / displacement loading ----------
def snapshot_field(
    model,
    mesh,
    T: float,
    steps: int = 500,
    device: torch.device | None = None,
):
    """
    Evaluate model over [0, T] and return verts_stack & tris as NumPy.

    Returns
    -------
    verts_stack : (steps, N_verts, D)
    tris        : (N_elems, 3)
    times       : (steps,)
    """
    if device is None:
        device = next(model.parameters()).device
    model = model.to(device)
    mesh = mesh.to(device)

    times = torch.linspace(0.0, T, steps=steps, device=device)

    model.eval()
    with torch.no_grad():
        disp, _, _ = model.eval_with_derivs(times)  # (steps, N_verts, 2)

    verts0 = mesh.verts_torch.to(device).detach()      # (N_verts, D)
    verts_stack = verts0[None, ...] + disp             # (steps, N_verts, D)

    tris = mesh.tris_torch.detach().cpu().numpy()      # (N_elems, 3)

    return (
        verts_stack.detach().cpu().numpy(),  # (T, N, D)
        tris,                                # (Ne, 3)
        times.detach().cpu().numpy(),        # (T,)
    )


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


def _find_disp_key(npz) -> str:
    """Guess displacement key in a .npz."""
    candidates = ["u", "disp", "displacement", "U", "verts_disp"]
    for k in candidates:
        if k in npz.files:
            return k
    raise KeyError(
        f"No displacement key found. Expected one of {candidates}, "
        f"found keys: {list(npz.files)}"
    )


def load_displacements_sequence(data_dir: Path, verts0: np.ndarray):
    """Load displacement time sequence and reconstruct verts_stack.

    Returns:
        verts_stack: (T_total, N, D)
    """
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    disp_chunks = []
    for f in npz_files:
        data = np.load(f)
        disp_key = _find_disp_key(data)
        disp = data[disp_key]

        # Normalize shape to (T, N, D)
        if disp.ndim == 2:  # (N, D)
            disp = disp[None, ...]  # (1, N, D)
        elif disp.ndim == 3:
            pass  # already (T, N, D)
        else:
            raise ValueError(
                f"Unexpected displacement shape {disp.shape} in {f}. "
                "Expected (N, D) or (T, N, D)."
            )

        # sanity check on N, D
        if disp.shape[1:] != verts0.shape:
            raise ValueError(
                f"Shape mismatch between disp {disp.shape[1:]} and "
                f"base verts {verts0.shape} in {f}."
            )

        disp_chunks.append(disp)

    disp_stack = np.concatenate(disp_chunks, axis=0)  # (T_total, N, D)
    verts_stack = verts0[None, ...] + disp_stack      # (T_total, N, D)

    return verts_stack, npz_files


# ---------- visualization: GIF ----------

def make_mesh_gif(verts_stack: np.ndarray, tris: np.ndarray,
                  out_path: Path, dpi: int = 150, fps: int = 15):
    """Create GIF of triangulated mesh over time."""
    T, N, D = verts_stack.shape
    assert D >= 2, "verts must have at least 2D coordinates."

    frames = []

    # Global bounds for consistent camera
    xs = verts_stack[..., 0]
    ys = verts_stack[..., 1]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)

    for t in range(T):
        verts = verts_stack[t]
        tri = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=dpi)
        ax.triplot(tri, linewidth=0.5)
        ax.set_aspect("equal", "box")
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.axis("off")

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf, dtype=np.uint8)[..., :3]
        frames.append(img)
        plt.close(fig)

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"[GIF] Saved: {out_path}")


# ---------- visualization: vibration heatmap ----------
def make_vibration_heatmap(verts_stack: np.ndarray, tris: np.ndarray,
                           out_path: Path, dpi: int = 1000):
    """Plot max vibration amplitude per node on reference configuration."""
    verts_ref = verts_stack[0]                  # (N, D)
    disp = verts_stack - verts_ref[None, ...]   # (T, N, D)
    amp = np.linalg.norm(disp, axis=-1)         # (T, N)
    amp_max = amp.max(axis=0)                   # (N,)

    tri = mtri.Triangulation(verts_ref[:, 0], verts_ref[:, 1], tris)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=dpi)
    tpc = ax.tripcolor(tri, amp_max, shading="gouraud")
    ax.set_aspect("equal", "box")
    ax.axis("off")

    cbar = fig.colorbar(tpc, ax=ax)
    cbar.set_label("Vibration amplitude (max |Δu|)")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] Saved vibration heatmap: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    cfg_json = load_config(args.config)
    run_name = cfg_json["training"]["name"]
    print(f"Start visualizing: {run_name}")

    device = args.device

    # Directories
    visuals_dir = Path("outputs/visuals")
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Load model + mesh from checkpoint
    model, design, aux, material, train_cfg = load_checkpoint(run_name, device=device)

    # Take field snapshots directly from model
    T = float(train_cfg.T)
    verts_stack, tris, times = snapshot_field(
        model=model,
        mesh=aux,
        T=T,
        steps=args.steps,
        device=torch.device(device),
    )

    print(
        f"Snapshot: steps={verts_stack.shape[0]}, "
        f"N_verts={verts_stack.shape[1]}, dim={verts_stack.shape[2]}"
    )

    # Outputs
    gif_path = visuals_dir / f"{run_name}.gif"
    png_path = visuals_dir / f"{run_name}.png"

    # Visualizations
    make_mesh_gif(verts_stack, tris, gif_path)
    make_vibration_heatmap(verts_stack, tris, png_path)


if __name__ == "__main__":
    main()