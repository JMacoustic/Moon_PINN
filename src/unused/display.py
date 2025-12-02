#!/usr/bin/env python3
import argparse, json, glob, os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import load_config


# ------------------------------- IO -------------------------------

def load_run(run_name: str):
    ckpt = Path("outputs/checkpoints") / run_name
    data = Path("outputs/data") / run_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}")
    if not data.exists():
        raise FileNotFoundError(f"Data dir not found: {data}")

    with open(ckpt / "train_config.json", "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    with open(ckpt / "latest_geometry.json", "r", encoding="utf-8") as f:
        geom = json.load(f)

    mesh_npz = np.load(ckpt / "latest_mesh.npz")
    V0 = mesh_npz["verts"].astype(np.float64)  # (nv,2)
    T  = mesh_npz["tris"].astype(np.int64)     # (nt,3)

    frames = []
    for fp in sorted(glob.glob(str(data / f"{run_name}_*.npz"))):
        arr = np.load(fp)
        t  = float(arr["t"])
        disp = arr["disp"].astype(np.float64)  # (nv,2)
        frames.append((t, disp))
    if not frames:
        raise FileNotFoundError(f"No snapshot npz files found in {data}")

    times = np.array([t for (t, _) in frames], dtype=np.float64)
    disps = np.stack([d for (_, d) in frames], axis=0)  # (ntimes, nv, 2)

    return V0, T, times, disps, train_cfg, geom


# -------------------------- Geometry helpers -----------------------

def compute_bottom_ids(V: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    y = V[:, 1]
    ymin = y.min()
    return np.nonzero(np.isclose(y, ymin, atol=atol))[0]


# ------------------------- Rendering helpers ------------------------

def fit_to_canvas(points: np.ndarray, W: int, H: int, pad: int = 20):
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    w = xmax - xmin; h = ymax - ymin
    if w <= 0: w = 1.0
    if h <= 0: h = 1.0
    sx = (W - 2*pad) / w
    sy = (H - 2*pad) / h
    s  = min(sx, sy)
    tx = pad - s * xmin + 0.5*(W - (2*pad + s*w))
    ty = pad - s * ymin + 0.5*(H - (2*pad + s*h))
    return s, tx, ty


def to_img_coords(P: np.ndarray, s: float, tx: float, ty: float, H: int):
    """World -> image coords (flip y for image)."""
    X = s * P[:, 0] + tx
    Y = s * P[:, 1] + ty
    Y = (H - Y)
    return np.stack([X, Y], axis=1)


def colorize(values: np.ndarray, vmin: float, vmax: float, cmap_name: str = "viridis") -> np.ndarray:
    cmap = plt.colormaps.get_cmap(cmap_name)
    v = (values - vmin) / max(vmax - vmin, 1e-12)
    v = np.clip(v, 0.0, 1.0)
    rgba = cmap(v)  # (n,4) in RGBA [0,1]
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    bgr = rgb[:, ::-1].copy()
    return bgr


def draw_tri_field(img: np.ndarray, V_img: np.ndarray, T: np.ndarray, tri_colors_bgr: np.ndarray):
    """Rasterize each triangle with a solid BGR color."""
    for k in range(T.shape[0]):
        tri = V_img[T[k]].astype(np.int32)
        cv2.fillConvexPoly(img, tri, color=tuple(int(c) for c in tri_colors_bgr[k]))


def draw_mesh_edges(img: np.ndarray, V_img: np.ndarray, T: np.ndarray, color=(30, 30, 30), thickness=1):
    edges = set()
    for a, b, c in T:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    for (i, j) in edges:
        p1 = tuple(V_img[i].astype(np.int32))
        p2 = tuple(V_img[j].astype(np.int32))
        cv2.line(img, p1, p2, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def legend_colorbar(img: np.ndarray, vmin: float, vmax: float, pos=(20, 20), size=(20, 200), cmap="viridis", label="|u| (m)"):
    x0, y0 = pos
    w, h = size
    bar = np.linspace(1, 0, h)[:, None] * np.ones((h, w))
    col = (plt.colormaps.get_cmap(cmap)(bar)[:, :, :3] * 255).astype(np.uint8)
    col = col[:, :, ::-1]  # to BGR
    y1 = y0 + h
    x1 = x0 + w
    img[y0:y1, x0:x1] = col
    # ticks
    for t, txt in [(0, vmax), (0.5, 0.5*(vmin+vmax)), (1, vmin)]:
        yy = int(y0 + t*h)
        cv2.line(img, (x1+4, yy), (x1+16, yy), (255, 255, 255), 1)
        cv2.putText(img, f"{txt:.2e}", (x1+20, yy+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(img, label, (x0, y0-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)


# ----------------------------- Pipelines ----------------------------

def make_video_disp(run_name: str, V0: np.ndarray, T: np.ndarray, times: np.ndarray, disps: np.ndarray,
                    out_mp4: Path, W: int = 900, H: int = 900, fps: int = 15, cmap="viridis"):
    """
    Color by instantaneous displacement magnitude |u|; triangle color = mean of its 3 vertices.
    """
    # global bounds for deformed bbox & |u| range (to reduce flicker)
    Vmin = V0.copy(); Vmax = V0.copy()
    mag_all = []
    for k in range(disps.shape[0]):
        U = disps[k]
        V = V0 + U
        Vmin = np.minimum(Vmin, V); Vmax = np.maximum(Vmax, V)
        mag = np.sqrt(np.sum(U**2, axis=1))              # (nv,)
        tri_mag = mag[T].mean(axis=1)                    # (nt,)
        mag_all.append(tri_mag)
    mag_all = np.stack(mag_all, axis=0)                  # (ntimes, nt)
    vmin = np.percentile(mag_all, 2.0)
    vmax = np.percentile(mag_all, 98.0)
    if vmax <= vmin: vmax = vmin + 1e-12

    # canvas fit
    s, tx, ty = fit_to_canvas(np.vstack([Vmin, Vmax]), W, H, pad=40)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_mp4), fourcc, fps, (W, H))

    for k, t in enumerate(times):
        U = disps[k]
        V = V0 + U
        Vimg = to_img_coords(V, s, tx, ty, H)

        tri_cols = colorize(mag_all[k], vmin, vmax, cmap_name=cmap)

        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_tri_field(frame, Vimg, T, tri_cols)
        draw_mesh_edges(frame, Vimg, T, color=(40, 40, 40), thickness=1)
        legend_colorbar(frame, vmin, vmax, pos=(20, 20), size=(24, 240), cmap=cmap, label="|u| (m)")
        cv2.putText(frame, f"t = {t:.3f} s", (W-220, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
        out.write(frame)

    out.release()


def make_rms_png(run_name: str, V0: np.ndarray, T: np.ndarray, times: np.ndarray, disps: np.ndarray,
                 out_png: Path, W: int = 1200, H: int = 900, cmap="viridis"):
    # per-vertex RMS amplitude (||u|| over time)
    amp = np.sqrt(np.mean(np.sum(disps**2, axis=2), axis=0))  # (nv,)

    # render on reference mesh with vertex color -> convert to per-triangle by averaging vertices
    tri_vals = amp[T].mean(axis=1)

    # mapping & colors
    vmin = np.percentile(amp, 2.0); vmax = np.percentile(amp, 98.0)
    if vmax <= vmin: vmax = vmin + 1e-12
    tri_cols = colorize(tri_vals, vmin, vmax, cmap_name=cmap)

    s, tx, ty = fit_to_canvas(V0, W, H, pad=60)
    Vimg = to_img_coords(V0, s, tx, ty, H)

    img = np.zeros((H, W, 3), dtype=np.uint8)
    draw_tri_field(img, Vimg, T, tri_cols)
    draw_mesh_edges(img, Vimg, T, color=(50, 50, 50), thickness=1)
    legend_colorbar(img, vmin, vmax, pos=(30, 30), size=(24, 300), cmap=cmap, label="RMS |u| (m)")
    cv2.putText(img, f"RMS displacement amplitude", (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_png), img)


def make_bottom_disp_plot(run_name: str, V0: np.ndarray, times: np.ndarray, disps: np.ndarray, out_png: Path):
    bottom_ids = compute_bottom_ids(V0, atol=1e-9)
    # mean absolute vertical displacement vs time
    mean_abs_v = np.mean(np.abs(disps[:, bottom_ids, 1]), axis=1)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(times, mean_abs_v, linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Mean |v_bottom| (m)")
    plt.title("Bottom-node vertical displacement (mean |v|)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# --------------------------------- Main --------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_json = load_config(args.config)

    NAME = cfg_json["training"]["name"]
    FPS = 15
    SIZE = [900, 900]
    CMAP = "viridis"
    OUTDIR = "outputs/visuals"

    V0, T, times, disps, train_cfg, geom = load_run(NAME)

    outdir = Path(OUTDIR); outdir.mkdir(parents=True, exist_ok=True)
    W, H = SIZE

    mp4_path = outdir / f"{NAME}_disp.mp4"
    rms_png  = outdir / f"{NAME}_rms.png"
    bot_png  = outdir / f"{NAME}_bottom_disp.png"

    print(f"[display] making displacement-colored video: {mp4_path}")
    make_video_disp(NAME, V0, T, times, disps, mp4_path, W=W, H=H, fps=FPS, cmap=CMAP)

    print(f"[display] making RMS displacement field png: {rms_png}")
    make_rms_png(NAME, V0, T, times, disps, rms_png, W=W, H=H, cmap=CMAP)

    print(f"[display] making bottom displacement plot: {bot_png}")
    make_bottom_disp_plot(NAME, V0, times, disps, bot_png)

    print("[display] done.")


if __name__ == "__main__":
    main()
