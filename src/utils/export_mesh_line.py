import numpy as np
from collections import defaultdict
from smooth_auxetic import *
import ezdxf
from utils import load_config, thickness_from_constraint
import argparse
import os

def _unique_edges_from_tris(tris: np.ndarray) -> np.ndarray:
    """Return sorted boundary edges (edges used by exactly one triangle)."""
    I = tris.astype(np.int64)
    e = np.concatenate(
        [I[:, [0, 1]],
         I[:, [1, 2]],
         I[:, [2, 0]]],
        axis=0,
    )
    e.sort(axis=1)  # undirected
    e_unique, counts = np.unique(e, axis=0, return_counts=True)
    return e_unique[counts == 1]   # boundary edges only


def _edge_loops(edges: np.ndarray) -> list[list[int]]:
    """
    Group undirected edges into ordered vertex loops / open chains.
    Assumes boundary graph with degree <= 2.
    """
    edges = np.asarray(edges, dtype=np.int64)
    n_edges = edges.shape[0]

    adj = defaultdict(list)  # vertex -> list[(neighbor, edge_idx)]
    for eidx, (a, b) in enumerate(edges):
        adj[a].append((b, eidx))
        adj[b].append((a, eidx))

    visited = np.zeros(n_edges, dtype=bool)
    loops: list[list[int]] = []

    for start_e in range(n_edges):
        if visited[start_e]:
            continue

        a, b = edges[start_e]
        loop = [a, b]
        visited[start_e] = True
        prev = a
        cur = b

        while True:
            nbrs = [(nbr, idx) for (nbr, idx) in adj[cur] if not visited[idx]]
            if not nbrs:
                break

            # avoid immediately going back if there is a choice
            nbr, idx = nbrs[0]
            if len(nbrs) > 1:
                for cand_nbr, cand_idx in nbrs:
                    if cand_nbr != prev:
                        nbr, idx = cand_nbr, cand_idx
                        break

            visited[idx] = True
            loop.append(nbr)
            prev, cur = cur, nbr

            # closed loop
            if nbr == loop[0]:
                break

        loops.append(loop)

    return loops


def export_edges_dxf_ezdxf(
    mesh: Aux,
    out_path: str = "auxetic_edges.dxf",
    layer: str = "MESH_EDGES",
    color: int = 7,
    z0: float = 0.0,
    scale: float = 1.0,
    origin=(0.0, 0.0),
    units: int = 6,
    quantize: float | None = None,
):
    """
    Export boundary edges as sequences of LINE entities arranged in loops,
    compatible with Abaqus DXF sketch import.
    """

    V = np.asarray(mesh._verts_np, dtype=float).copy()
    T = np.asarray(mesh._tris_np, dtype=np.int64)
    E = _unique_edges_from_tris(T)
    loops = _edge_loops(E)

    if quantize is not None and quantize > 0:
        V = np.round(V / quantize) * quantize

    V *= float(scale)
    V[:, 0] += float(origin[0])
    V[:, 1] += float(origin[1])

    # R12 is often safest for FEA imports
    doc = ezdxf.new("R12")
    if units is not None:
        doc.header["$INSUNITS"] = int(units)

    if layer not in doc.layers:
        doc.layers.add(layer, color=color)

    msp = doc.modelspace()

    # Emit LINE entities in loop order
    n_lines = 0
    for loop in loops:
        if len(loop) < 2:
            continue

        is_closed = (loop[0] == loop[-1])
        # if closed and last == first, skip the duplicate for nicer logic
        if is_closed and len(loop) > 1:
            idx_seq = loop[:-1]
        else:
            idx_seq = loop

        # connect each vertex to the next
        for i in range(len(idx_seq) - 1):
            a = idx_seq[i]
            b = idx_seq[i + 1]
            xa, ya = V[a, 0], V[a, 1]
            xb, yb = V[b, 0], V[b, 1]
            msp.add_line((xa, ya, z0), (xb, yb, z0),
                         dxfattribs={"layer": layer, "color": color})
            n_lines += 1

        # close loop explicitly for closed contours
        if is_closed and len(idx_seq) > 1:
            a = idx_seq[-1]
            b = idx_seq[0]
            xa, ya = V[a, 0], V[a, 1]
            xb, yb = V[b, 0], V[b, 1]
            msp.add_line((xa, ya, z0), (xb, yb, z0),
                         dxfattribs={"layer": layer, "color": color})
            n_lines += 1

    doc.saveas(out_path)
    print(f"[DXF saved] {out_path}  (loops={len(loops)}, lines={n_lines})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg_json = load_config(args.config)

    # ----- geometry -----
    C = cfg_json["geometry"]["C"]
    grid_size = tuple(cfg_json["geometry"]["grid_size"])
    pitch = tuple(cfg_json["geometry"]["pitch"])
    x_offset = cfg_json["geometry"]["x_offset"]
    thickness = thickness_from_constraint(C, pitch[0], pitch[1], x_offset)

    # path
    run_name = cfg_json["training"]["name"]
    out_dir = os.path.join("outputs", "dxf")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_name}.dxf")

    Mesh = Aux(
        grid_size=grid_size,
        pitch=pitch,
        x_offset=x_offset,
        thickness=thickness,
        add_diagonals=False,
        quantize=1e-9,
        joint_radius=None,
        joint_cap="square",
        remove_horizontal_rule="checker_even",
        keep_boundary_horiz=True
    )

    export_edges_dxf_ezdxf(Mesh, out_path)



#         Mesh = Aux(
#         grid_size=(5, 20),
#         pitch = (0.1362, 0.1034),
#         x_offset=0.033,
#         thickness=0.00978,
#         add_diagonals=False,
#         quantize=1e-9,
#         joint_radius=None,
#         joint_cap="square",
#         remove_horizontal_rule="checker_even",
#         keep_boundary_horiz=True
# )