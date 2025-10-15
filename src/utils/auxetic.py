import numpy as np
import triangle as tr
import matplotlib.pyplot as plt

import numpy as np
import triangle as tr

import math
import numpy as np
from math import hypot, atan2
import numpy as np, math
from math import hypot

def generate_auxetic(
    grid_size=(10, 10),
    pitch=(1.0, 1.0),
    x_offset=0.2,
    thickness=0.15,
    add_diagonals=False,
    quantize=1e-9,
    joint_radius=None,         # interpreted here as HALF side-length cap if given; else = 0.5*thickness
    joint_cap="square",        # kept for API; always builds small squares (use 'none' to skip caps)
    remove_horizontal_rule="checker_even",  # 'none' | 'checker_even' (remove when (i+j)%2==0)
    keep_boundary_horiz=True,  # keep left/right boundary horizontals even if rule removes
):
    """
    Returns:
        {
          'verts'  : flat [x0,y0, x1,y1, ...],
          'triIds' : flat [i0,i1,i2,  i3,i4,i5, ...],
          'edgeIds': [(a,b), ...] from triangles,
          'bcIds'  : [indices of vertices on the top boundary (max y)],
          'lattice_points': P  # (ny,nx,2)
        }
    """
    nx, ny = grid_size
    px, py = pitch
    t = float(thickness)
    # half side-length of the joint square
    s = (0.5 * t) if (joint_radius is None) else float(joint_radius)
    s = max(1e-12, min(s, 0.5 * t))  # be conservative: square ≤ beam width

    # --- 1) Re-entrant lattice points (with ±x shift) ---
    P = np.zeros((ny, nx, 2), dtype=float)
    for j in range(ny):
        for i in range(nx):
            sign = -1.0 if ((i + j) % 2 == 1) else 1.0
            P[j, i] = (i * px + sign * x_offset, j * py)

    # --- helpers to build a deduped triangulation ---
    verts, triIds, vmap = [], [], {}
    def add_vertex(v):
        key = (round(v[0] / quantize), round(v[1] / quantize))
        idx = vmap.get(key)
        if idx is None:
            idx = len(verts)
            vmap[key] = idx
            verts.append([v[0], v[1]])
        return idx

    def add_tri(a, b, c):
        triIds.extend([a, b, c])

    def add_quad(a, b, c, d):
        # triangles: (a,b,c) + (a,c,d)
        ia, ib, ic, id_ = map(add_vertex, (a, b, c, d))
        add_tri(ia, ib, ic)
        add_tri(ia, ic, id_)

    # --- 2) Build a tiny square joint at every node ---
    # Store corners and side segments for each node for easy linkage
    # Corners (TL, TR, BR, BL). Sides: 'L'=(TL,BL), 'R'=(TR,BR), 'T'=(TL,TR), 'B'=(BL,BR)
    joints = {}
    if joint_cap != "none":
        for j in range(ny):
            for i in range(nx):
                cx, cy = P[j, i]
                TL = (cx - s, cy + s)
                TR = (cx + s, cy + s)
                BR = (cx + s, cy - s)
                BL = (cx - s, cy - s)
                joints[(j, i)] = {
                    "TL": TL, "TR": TR, "BR": BR, "BL": BL,
                    "L": (TL, BL), "R": (TR, BR), "T": (TL, TR), "B": (BL, BR)
                }
                # add the cap as two triangles (you can comment this out if you only want implicit joints)
                add_quad(TL, TR, BR, BL)
    else:
        # still define corners/sides (degenerate size) so linkage code works;
        # if joint_cap=='none', set s very small but non-zero.
        s_eps = 1e-9
        for j in range(ny):
            for i in range(nx):
                cx, cy = P[j, i]
                TL = (cx - s_eps, cy + s_eps)
                TR = (cx + s_eps, cy + s_eps)
                BR = (cx + s_eps, cy - s_eps)
                BL = (cx - s_eps, cy - s_eps)
                joints[(j, i)] = {
                    "TL": TL, "TR": TR, "BR": BR, "BL": BL,
                    "L": (TL, BL), "R": (TR, BR), "T": (TL, TR), "B": (BL, BR)
                }

    # --- 3) Linkages: each beam uses ONE side from each adjacent joint square ---
    # Horizontal linkage: right side of (j,i) to left side of (j,i+1)
    def keep_horizontal(i, j):
        if remove_horizontal_rule == "none":
            return True
        if keep_boundary_horiz and (j == 0 or j == ny-1 ):
            return True
        # Auxetic “checker” removal makes re-entrant look:
        # remove when (i+j) is even (per your latest rule)
        return ((i + j) % 2 != 0)

    for j in range(ny):
        for i in range(nx - 1):
            if not keep_horizontal(i, j):
                continue
            left = joints[(j, i)]
            right = joints[(j, i + 1)]
            # Build a clean quad: [left.R top → left.R bottom → right.L bottom → right.L top]
            A = left["R"][0]   # left TR
            B = left["R"][1]   # left BR
            C = right["L"][1]  # right BL
            D = right["L"][0]  # right TL
            add_quad(A, B, C, D)

    # Vertical linkage: top side of (j,i) to bottom side of (j+1,i)
    for j in range(ny - 1):
        for i in range(nx):
            lower = joints[(j, i)]
            upper = joints[(j + 1, i)]
            # Quad: [lower.T right → lower.T left → upper.B left → upper.B right]
            A = lower["T"][1]  # lower TR
            B = lower["T"][0]  # lower TL
            C = upper["B"][0]  # upper BL
            D = upper["B"][1]  # upper BR
            add_quad(A, B, C, D)

    # Optional diagonals (each uses one side from each touching joint)
    if add_diagonals:
        # NE diagonals: use (j,i) top-right corner side to (j+1,i+1) bottom-left side
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = joints[(j, i)]
                b = joints[(j + 1, i + 1)]
                # use the short “corner sides”: right of top node to left of bottom-right node
                A = a["R"][0]   # near TR of a
                B = a["T"][1]   # TR of a (same point as above, but keeps pattern consistent)
                C = b["L"][0]   # TL of b
                D = b["B"][0]   # BL of b
                add_quad(A, B, C, D)

        # NW diagonals: mirror
        for j in range(ny - 1):
            for i in range(1, nx):
                a = joints[(j, i)]
                b = joints[(j + 1, i - 1)]
                A = a["T"][0]   # TL of a
                B = a["L"][0]   # TL of a (same)
                C = b["B"][1]   # BR of b
                D = b["R"][1]   # BR of b (same)
                add_quad(A, B, C, D)

    # --- edges & boundary ids for convenience ---
    verts_arr = np.asarray(verts, dtype=float)
    tri_arr = np.asarray(triIds, dtype=int).reshape(-1, 3) if triIds else np.zeros((0, 3), dtype=int)

    edges = set()
    for a, b, c in tri_arr:
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    edgeIds = sorted(edges)

    if len(verts_arr) > 0:
        y_max = float(verts_arr[:, 1].max())
        bcIds = np.where(np.abs(verts_arr[:, 1] - y_max) < 1e-9)[0].tolist()
    else:
        bcIds = []

    return {
        "verts": verts_arr.flatten().tolist(),
        "triIds": tri_arr.flatten().tolist(),
        "edgeIds": edgeIds,
        "bcIds": bcIds,
        "lattice_points": P,
    }

import numpy as np
from math import sqrt

def _infer_px_xoff(P):
    """Infer (px, x_offset) from lattice points along a row with nx>=3."""
    ny, nx, _ = P.shape
    j = 0
    xs = [P[j,i,0] for i in range(nx)]
    diffs = [xs[i+1]-xs[i] for i in range(nx-1)]
    # diffs alternate: px-2xoff, px+2xoff, ...
    if len(diffs) < 2:
        # fallback: use mean spacing, assume x_offset ~ 0
        return (np.mean(diffs) if diffs else 1.0, 0.0)
    d0, d1 = diffs[0], diffs[1]
    px = 0.5*(d0 + d1)
    xoff = 0.25*abs(d1 - d0)
    return px, xoff

def _estimate_s_from_vertices(P, verts_arr):
    """Estimate joint half-cap size s from the nearest vertex distance to a center."""
    ny, nx, _ = P.shape
    # pick a middle-ish node to avoid boundaries
    j = min(1, ny-1-1) if ny>=3 else 0
    i = min(1, nx-1-1) if nx>=3 else 0
    cx, cy = P[j,i]
    V = verts_arr
    d2 = (V[:,0]-cx)**2 + (V[:,1]-cy)**2
    k = int(np.argmin(d2))
    dist = sqrt(d2[k])
    # nearest corner is at (cx±s, cy±s), distance = sqrt( (±s)^2 + (±s)^2 ) = sqrt(2)*s
    s = max(1e-12, dist / sqrt(2))
    return s

def _build_corner_index(P, verts_arr, s, quant=1e-9):
    """
    Return dict[(j,i)] -> {'TL':vidx,'TR':vidx,'BR':vidx,'BL':vidx}
    by hashing rounded coordinates.
    """
    vmap = {}
    for idx,(x,y) in enumerate(verts_arr):
        key = (round(x/quant), round(y/quant))
        vmap[key] = idx

    ny, nx, _ = P.shape
    corner_idx = {}
    for j in range(ny):
        for i in range(nx):
            cx, cy = P[j,i]
            TL = (cx - s, cy + s)
            TR = (cx + s, cy + s)
            BR = (cx + s, cy - s)
            BL = (cx - s, cy - s)
            keys = [ (round(TL[0]/quant),round(TL[1]/quant)),
                     (round(TR[0]/quant),round(TR[1]/quant)),
                     (round(BR[0]/quant),round(BR[1]/quant)),
                     (round(BL[0]/quant),round(BL[1]/quant)) ]
            # tolerate rare rounding misses by nearest search fallback
            vids = []
            for kx in keys:
                if kx in vmap:
                    vids.append(vmap[kx])
                else:
                    # nearest fallback
                    target = np.array([kx[0]*quant, kx[1]*quant])
                    d2 = np.sum((verts_arr - target)**2, axis=1)
                    vids.append(int(np.argmin(d2)))
            corner_idx[(j,i)] = {'TL': vids[0], 'TR': vids[1], 'BR': vids[2], 'BL': vids[3]}
    return corner_idx

def _make_lattice(P, px_new, py_new, xoff_new):
    """Recompute lattice centers P' with updated pitch/x_offset, preserving sign pattern."""
    ny, nx, _ = P.shape
    Pp = np.zeros_like(P)
    # recover the original sign pattern from P (based on (i+j)%2)
    # sign = -1 if (i+j) odd else +1 (matches your generator)
    for j in range(ny):
        for i in range(nx):
            sign = -1.0 if ((i + j) % 2 == 1) else 1.0
            Pp[j,i,0] = i*px_new + sign*xoff_new
            Pp[j,i,1] = j*py_new
    return Pp

def mutate_aux_inplace(
    aux,
    pitch=None,           # (px_new, py_new) or None
    x_offset=None,        # float or None
    thickness=None,       # float or None
    keep_meta=True,       # store mapping to speed future updates
    quantize=1e-9
):
    """
    In-place update of aux['verts'] for new pitch / x_offset / thickness.
    Keeps triIds/edgeIds as is. Recomputes bcIds.
    Requirements:
      - aux has 'lattice_points' (P).
      - The mesh is built only from joint-corner vertices (which your generator does).
    If 'meta' with 'corner_idx' and 's' is absent, it reconstructs them once.
    """
    # ---- fetch arrays ----
    verts = np.asarray(aux['verts'], dtype=float).reshape(-1,2)
    P = np.asarray(aux['lattice_points'], dtype=float)  # (ny,nx,2)
    ny, nx, _ = P.shape

    # ---- infer old parameters if needed ----
    px_old, xoff_old = _infer_px_xoff(P)
    # py is simple: row spacing
    if ny >= 2:
        py_old = float(np.mean(P[1:,0,1] - P[:-1,0,1]))
    else:
        py_old = 1.0

    # ---- 0) quick path: pitch-only change with uniform scaling ----
    # If only pitch is requested (x_offset & thickness unchanged/None),
    # we can do a global affine scale: x*=sx, y*=sy (this also scales x_offset implicitly).
    if pitch is not None and x_offset is None and thickness is None and 'meta' not in aux:
        px_new, py_new = pitch
        sx = (px_new / px_old) if px_old != 0 else 1.0
        sy = (py_new / py_old) if py_old != 0 else 1.0
        verts[:,0] *= sx
        verts[:,1] *= sy
        aux['verts'] = verts.flatten().tolist()
        # bcIds recompute
        y_max = float(verts[:,1].max()) if len(verts) else -np.inf
        aux['bcIds'] = np.where(np.abs(verts[:,1]-y_max) < 1e-9)[0].tolist()
        # also update lattice_points (P)
        P_scaled = P.copy()
        P_scaled[:,:,0] *= sx
        P_scaled[:,:,1] *= sy
        aux['lattice_points'] = P_scaled
        return aux

    # ---- ensure meta ----
    meta = aux.get('meta', {})
    if 'corner_idx' not in meta or 's' not in meta:
        s_est = _estimate_s_from_vertices(P, verts)
        corner_idx = _build_corner_index(P, verts, s_est, quant=quantize)
        meta.update({'corner_idx': corner_idx, 's': s_est})
        if keep_meta:
            aux['meta'] = meta

    corner_idx = meta['corner_idx']
    s_old = float(meta['s'])

    # ---- choose new parameters ----
    px_new, py_new = (pitch if pitch is not None else (px_old, py_old))
    xoff_new = float(x_offset) if x_offset is not None else xoff_old
    if thickness is not None:
        t_new = float(thickness)
        s_new = max(1e-12, min(0.5*t_new, 0.5*t_new))  # clamp like generator
    else:
        # keep the same corner size proportion relative to old thickness
        s_new = s_old

    # ---- recompute lattice centers with new pitch/x_offset ----
    P_new = _make_lattice(P, px_new, py_new, xoff_new)

    # ---- write new corner coordinates into verts, in place ----
    for j in range(ny):
        for i in range(nx):
            cx, cy = P_new[j,i]
            TL = (cx - s_new, cy + s_new)
            TR = (cx + s_new, cy + s_new)
            BR = (cx + s_new, cy - s_new)
            BL = (cx - s_new, cy - s_new)
            idxs = corner_idx[(j,i)]
            verts[idxs['TL']] = TL
            verts[idxs['TR']] = TR
            verts[idxs['BR']] = BR
            verts[idxs['BL']] = BL

    # ---- commit arrays & update bcIds ----
    aux['verts'] = verts.flatten().tolist()
    aux['lattice_points'] = P_new
    y_max = float(verts[:,1].max()) if len(verts) else -np.inf
    aux['bcIds'] = np.where(np.abs(verts[:,1]-y_max) < 1e-9)[0].tolist()

    # keep triIds/edgeIds as-is (topology unchanged)
    # meta: store updated s for future quick edits
    meta['s'] = s_new
    if keep_meta:
        aux['meta'] = meta
    return aux



# Test and plot
if __name__ == "__main__":
    mesh = generate_auxetic()
    vertices = np.array(mesh['verts']).reshape(-1, 2)
    triangles = np.array(mesh['triIds']).reshape(-1, 3)
    bc_ids = mesh['bcIds']

    plt.figure(figsize=(6, 6))
    for tri in triangles:
        pts = vertices[tri]
        plt.fill(*zip(*pts), edgecolor='black', fill=False)

    plt.plot(vertices[:, 0], vertices[:, 1], 'ko', markersize=2, label="All triangular components")
    plt.plot(vertices[bc_ids, 0], vertices[bc_ids, 1], 'ro', markersize=4, label="BC nodes")

    plt.axis('equal')
    plt.legend(loc="lower right")
    plt.title('Auxetic Mesh')
    plt.show()



# def generate_mesh(outer_size=0.5, inner_size=0.3, hole_center=(0.0, 0.0), grid_size=(3, 3)):
#     points = []
#     segments = []
#     holes = []

#     tile_width = outer_size
#     tile_height = outer_size

#     point_offset = 0
#     for i in range(grid_size[0]):
#         for j in range(grid_size[1]):
#             offset_x = i * tile_width
#             offset_y = j * tile_height

#             # Outer square (counter-clockwise)
#             w_outer = outer_size / 2
#             outer = [
#                 [offset_x - w_outer, offset_y - w_outer],
#                 [offset_x + w_outer, offset_y - w_outer],
#                 [offset_x + w_outer, offset_y + w_outer],
#                 [offset_x - w_outer, offset_y + w_outer]
#             ]
#             outer_segments = [[point_offset + k, point_offset + (k + 1) % 4] for k in range(4)]

#             # Inner square (clockwise)
#             w_inner = inner_size / 2
#             cx, cy = hole_center
#             cx += offset_x
#             cy += offset_y
#             inner = [
#                 [cx - w_inner, cy - w_inner],
#                 [cx + w_inner, cy - w_inner],
#                 [cx + w_inner, cy + w_inner],
#                 [cx - w_inner, cy + w_inner]
#             ]
#             inner_segments = [[point_offset + 4 + k, point_offset + 4 + (k + 1) % 4] for k in range(4)]

#             points.extend(outer + inner)
#             segments.extend(outer_segments + inner_segments)
#             holes.append([cx, cy])
#             point_offset += 8

#     A = dict(vertices=np.array(points),
#              segments=np.array(segments),
#              holes=np.array(holes))

#     try:
#         B = tr.triangulate(A, 'p q')
#         if 'vertices' not in B or 'triangles' not in B:
#             raise RuntimeError("Triangulation failed or returned incomplete result")
#     except Exception as e:
#         print("Triangulation error:", e)
#         return None

#     if 'vertices' not in B or 'triangles' not in B:
#         raise RuntimeError("Triangulation failed!")

#     verts = B['vertices'].flatten().tolist()
#     triIds = B['triangles'].flatten().tolist()

#     # Build boundary edges
#     edges_set = set()
#     for tri in B['triangles']:
#         for i in range(3):
#             a, b = tri[i], tri[(i + 1) % 3]
#             edge = tuple(sorted((a, b)))
#             edges_set.add(edge)

#     edges_list = list(edges_set)

#     verts_array = np.array(verts).reshape(-1, 2)
#     y_max = np.max(verts_array[:, 1])
#     tol = 1e-6
#     bc_mask = np.abs(verts_array[:, 1] - y_max) < tol
#     bc_ids = np.where(bc_mask)[0].tolist()

#     return {
#         'verts': verts,
#         'triIds': triIds,
#         'edgeIds': edges_list,
#         'bcIds' : bc_ids
#     }