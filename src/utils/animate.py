import re, glob, numpy as np, matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.animation import FFMpegWriter, PillowWriter
from pathlib import Path
from src.utils.auxetic import *

def animate(cfg, SCALE=1.5, FPS=15, DPI=150, LINE_W=0.6):
    # ----------------- Config -----------------
    SNAP_GLOB = f"outputs/data/{cfg.name}_*.npz"     # matches ..._{k:04d}.npz
    MP4_PATH  = f"outputs/video/{cfg.name}.mp4"
    GIF_PATH  = f"outputs/gif/{cfg.name}.gif"

    Path("outputs/video").mkdir(parents=True, exist_ok=True)
    Path("outputs/gif").mkdir(parents=True, exist_ok=True)

    # -------------- Load mesh (verts/tris) --------------
    AUX = generate_auxetic()
    verts = np.asarray(AUX["verts"], dtype=float).reshape(-1, 2)
    tri   = np.asarray(AUX["triIds"], dtype=int).reshape(-1, 3)
    triang = Triangulation(verts[:,0], verts[:,1], tri)

    # -------------- Gather snapshot frames --------------
    files = sorted(glob.glob(SNAP_GLOB))
    if not files:
        raise SystemExit(f"No snapshot files found at {SNAP_GLOB}. Train first or check the path.")

    _num = re.compile(r"_(\d{4})\.npz$")
    files.sort(key=lambda p: int(_num.search(p).group(1)) if _num.search(p) else 0)

    frames, times = [], []
    for p in files:
        z = np.load(p)
        times.append(float(z["t"]))
        frames.append(np.asarray(z["disp"], dtype=float))  # (N,2)

    frames = np.stack(frames, axis=0)  # (T, N, 2)
    T, N, _ = frames.shape
    assert N == verts.shape[0], "Snapshot vertex count mismatches mesh verts."

    # -------------- Figure setup --------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Auxetic PINN displacement (deformed mesh)')

    pad = 0.05 * (verts.max() - verts.min())
    ax.set_xlim(verts[:,0].min()-pad, verts[:,0].max()+pad)
    ax.set_ylim(verts[:,1].min()-pad, verts[:,1].max()+pad)

    disp0 = SCALE * frames[0]
    coords0 = verts + disp0
    tri_lines = ax.triplot(coords0[:,0], coords0[:,1], tri, linewidth=LINE_W)
    text_time = ax.text(0.02, 0.98, f"t = {times[0]:.3f}", transform=ax.transAxes, va='top', ha='left')

    def _update(k):
        disp = SCALE * frames[k]
        coords = verts + disp
        # remove old and draw new lines
        for ln in tri_lines:
            ln.remove()
        new_lines = ax.triplot(coords[:,0], coords[:,1], tri, linewidth=LINE_W)
        tri_lines[:] = new_lines
        text_time.set_text(f"t = {times[k]:.3f}")
        return (*tri_lines, text_time)

    # # -------------- Export MP4 --------------
    # try:
    #     writer = FFMpegWriter(fps=FPS, bitrate=2400)
    #     with writer.saving(fig, MP4_PATH, DPI):
    #         for k in range(T):
    #             _update(k)
    #             writer.grab_frame()
    #     print(f"Saved MP4 to {MP4_PATH}")
    # except Exception as e:
    #     print("[WARN] MP4 export failed (ffmpeg missing?). Skipping.", e)

    # -------------- Export GIF --------------
    try:
        writer = PillowWriter(fps=FPS)
        with writer.saving(fig, GIF_PATH, DPI):
            for k in range(T):
                _update(k)
                writer.grab_frame()
        print(f"Saved GIF to {GIF_PATH}")
    except Exception as e:
        print("[WARN] GIF export failed:", e)

    plt.close(fig)

