# visualize_vibration.py
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import imageio.v2 as imageio
from utils import load_config


# ---------- mesh / displacement loading ----------

def load_base_mesh(run_name: str):
    """Load reference verts/tris from latest_mesh.npz."""
    ckpt_dir = Path("outputs/checkpoints") / run_name
    mesh_path = ckpt_dir / "latest_mesh.npz"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"{mesh_path} not found.\n"
            "Make sure Checkpointer.save_state() was called with tag='latest' "
            "and that it saved 'latest_mesh.npz'."
        )

    data = np.load(mesh_path)
    if "verts" not in data or "tris" not in data:
        raise KeyError(
            f"{mesh_path} must contain 'verts' and 'tris' arrays."
        )

    verts0 = data["verts"]  # (N, D)
    tris = data["tris"]     # (M, 3)
    return verts0, tris


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
    args = parser.parse_args()

    cfg_json = load_config(args.config)

    run_name = cfg_json["training"]["name"]
    print(f"Start visualizing: {run_name}")

    # Directories
    data_dir = Path("outputs/data") / run_name
    visuals_dir = Path("outputs/visuals")
    visuals_dir.mkdir(parents=True, exist_ok=True)

    # Load mesh + displacements
    verts0, tris = load_base_mesh(run_name)
    verts_stack, npz_files = load_displacements_sequence(data_dir, verts0)
    # npz = np.load("outputs/data/251127_final_soft_simonly_v2/251127_final_soft_simonly_v2_0010.npz")
    # print(npz["disp"])
    print(f"Loaded {len(npz_files)} displacement files, "
          f"total frames: {verts_stack.shape[0]}")

    # Outputs
    gif_path = visuals_dir / f"{run_name}.gif"
    png_path = visuals_dir / f"{run_name}.png"

    # Visualizations
    make_mesh_gif(verts_stack, tris, gif_path)
    make_vibration_heatmap(verts_stack, tris, png_path)


if __name__ == "__main__":
    main()