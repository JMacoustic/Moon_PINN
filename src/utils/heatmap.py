import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import glob, re
from typing import Optional
from auxetic import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import glob, re
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import glob, re
from typing import Optional

# Expect these to exist in your environment
# - generate_auxetic(): returns dict with keys "verts" (flat), "triIds" (flat)
# - cfg: object with a .name attribute


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import glob, re
from typing import Optional

# Expect these to exist in your environment
# - generate_auxetic(): returns dict with keys "verts" (flat), "triIds" (flat)
# - cfg: object with a .name attribute


def plot_max_amp_heatmap(cfg, *, SAVE_PNG=True, OUT_DIR="outputs/figs", ALPHA=0.9,
                         CLIP_Q: Optional[float] = 0.0,
                         CMAP: str = "viridis",
                         EDGE_W: float = 0.4,
                         TITLE: Optional[str] = None,
                         INVERT_VALUES: bool = True):
    """
    Draws the maximum vibration amplitude heatmap over the mesh structure.

    - Loads time snapshots from outputs/data/{cfg.name}_*.npz where each file has arrays:
        t     : scalar time
        disp  : shape (N,2) per-vertex displacement
    - Computes per-vertex max amplitude over time: max_t ||u_i(t)||_2
    - Overlays a colored heatmap (per-vertex) on top of the wireframe mesh.
    - If `INVERT_VALUES=True`, flips the scalar field so that high↔low colors are swapped while keeping geometry unchanged.

    Args:
        cfg: object with attribute `name` matching snapshot prefix.
        SAVE_PNG: save figure as PNG under OUT_DIR.
        OUT_DIR: directory to save the PNG.
        ALPHA: heatmap opacity (0..1), higher = less transparent.
        CLIP_Q: optional upper clipping quantile in [0,1). E.g., 0.98 to clip top 2% outliers.
        CMAP: matplotlib colormap name.
        EDGE_W: wireframe line width.
        TITLE: optional title; defaults to f"Max amplitude — {cfg.name}".
    """
    name = cfg.name
    SNAP_GLOB = f"outputs/data/{name}_*.npz"

    # ---------------- Mesh ----------------
    AUX = generate_auxetic()
    verts = np.asarray(AUX["verts"], dtype=float).reshape(-1, 2)
    tri   = np.asarray(AUX["triIds"], dtype=int).reshape(-1, 3)
    triang = Triangulation(verts[:, 0], verts[:, 1], tri)

    # ---------------- Load snapshots ----------------
    files = sorted(glob.glob(SNAP_GLOB))
    if not files:
        raise SystemExit(f"No snapshot files found at {SNAP_GLOB}. Train first or check the path.")

    _num = re.compile(r"_(\d{4})\\.npz$")
    files.sort(key=lambda p: int(_num.search(p).group(1)) if _num.search(p) else 0)

    amps = None  # will accumulate max per-vertex amplitude
    for p in files:
        z = np.load(p)
        disp = np.asarray(z["disp"], dtype=float)  # (N,2)
        if disp.shape[0] != verts.shape[0]:
            raise ValueError(f"Vertex count mismatch: disp has {disp.shape[0]} vs verts {verts.shape[0]} in {p}")
        a = np.linalg.norm(disp, axis=1)  # (N,)
        amps = a if amps is None else np.maximum(amps, a)

    # Optional clipping to improve visual range (ignore extreme outliers)
    vmin = float(np.min(amps))
    vmax = float(np.max(amps))
    if CLIP_Q and 0.0 < CLIP_Q < 1.0:
        vmax = float(np.quantile(amps, CLIP_Q))

    # ---------------- Plot ----------------
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Choose which field to plot (optionally flipped around the mid-range)
    _vmin_plot = vmin
    _vmax_plot = vmax
    amps_plot = amps
    if INVERT_VALUES:
        # flip scalar field but keep the same colorbar range so mesh stays put, colors invert
        amps_plot = (vmin + vmax) - amps

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.set_aspect('equal', adjustable='box')

    pad = 0.05 * (verts.max() - verts.min())
    ax.set_xlim(verts[:, 0].min() - pad, verts[:, 0].max() + pad)
    ax.set_ylim(verts[:, 1].min() - pad, verts[:, 1].max() + pad)

    # Per-vertex coloring; use 'gouraud' to smoothly interpolate across triangles
    tpc = ax.tripcolor(triang, amps_plot, shading='gouraud', cmap=CMAP, vmin=_vmin_plot, vmax=_vmax_plot, alpha=ALPHA, zorder=1)

    # Wireframe on top
    ax.triplot(triang, linewidth=EDGE_W, color='k', alpha=0.9, zorder=2)

    cb = fig.colorbar(tpc, ax=ax, fraction=0.046, pad=0.04)

    # Robust colorbar label: try mathtext, fallback to plain text if parsing fails
    _math_label  = r"$\max_t \| \mathbf{u}(t) \|_2$ (displacement amplitude)"
    _plain_label = "max_t ||u(t)||_2 (displacement amplitude)"
    try:
        cb.set_label(_math_label)
    except Exception:
        cb.set_label(_plain_label)

    ttl = TITLE or f"Max amplitude — {name}"
    ax.set_title(ttl)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    out_path = Path(OUT_DIR) / f"{name}_max_amp_heatmap.png"
    if SAVE_PNG:
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")

    return fig, ax, {
        "amps": amps,
        "amps_plot": amps_plot,
        "vmin": _vmin_plot,
        "vmax": _vmax_plot,
        "inverted": bool(INVERT_VALUES),
        "out_png": str(out_path)
    }


# ---------------- Example usage ----------------
if __name__ == "__main__":
    class Cfg: name = "251001_testrun_v0"
    plot_max_amp_heatmap(Cfg(), CLIP_Q=0.98)
