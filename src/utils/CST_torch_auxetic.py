import numpy as np
import torch

import numpy as np
import torch
import torch.nn as nn
from utils.utils import *

class Aux(nn.Module):
    def __init__(
        self,
        grid_size=(10, 10),
        pitch=(1.0, 1.0),
        x_offset=0.2,
        C_constraint = 0.004,
        add_diagonals=False,
        quantize=1e-9,
        joint_cap="square",
        remove_horizontal_rule="checker_even",
        keep_boundary_horiz=True,
    ) -> None:
        super().__init__()

        # --- constants / non-trainable config ---
        self.grid_size = tuple(grid_size)
        self.add_diagonals = bool(add_diagonals)
        self.quantize = float(quantize)
        self.joint_cap = joint_cap
        self.remove_horizontal_rule = remove_horizontal_rule
        self.keep_boundary_horiz = bool(keep_boundary_horiz)

        # frozen copies used only for initial NumPy topology generation
        self._pitch0 = tuple(pitch)
        self._x_offset0 = float(x_offset)

        px0, py0 = self._pitch0
        xoff0 = self._x_offset0
 
        self._thickness0 = thickness_from_constraint(C_constraint, px0, py0, xoff0)

        # --- trainable design parameters (used in differentiable path) ---
        self.pitch = nn.Parameter(torch.tensor(pitch, dtype=torch.float32))  # (2,)
        self.x_offset = nn.Parameter(torch.tensor(float(x_offset), dtype=torch.float32))
        
        # Outputs / buffers
        self.register_buffer("C_constraint", torch.tensor(C_constraint, dtype=torch.float32))
        self.register_buffer("verts_torch", torch.empty(0, 2, dtype=torch.float32))
        self.register_buffer("tris_torch",  torch.empty(0, 3, dtype=torch.int64))
        self.register_buffer("bottom_ids",  torch.empty(0,    dtype=torch.int64))
        self.register_buffer("top_ids",     torch.empty(0,    dtype=torch.int64))
        self.nv = 0
        self.nt = 0
        self.alias_probs = None

        self.joints = {}
        self.register_buffer("_corner_ref", torch.empty(0, 3, dtype=torch.long))

        self._generate_mesh_vectorized()
        self._generate_torch_mesh()
        self._compute_boundary_ids()

    @property
    def thickness(self) -> torch.Tensor:
        """
        Current thickness as a 0-D tensor, determined by the constraint:
            thickness = thickness_from_constraint_torch(C, px, py, xoff).
        This is differentiable w.r.t. pitch and x_offset.
        """
        px, py = self.pitch  # (2,) tensor → each is 0-D
        return thickness_from_constraint_torch(self.C_constraint, px, py, self.x_offset)


    def _keep_horizontal_mask(self, nx, ny):
        """Boolean mask for horizontal quads between (j,i) and (j,i+1). Shape (ny, nx-1)."""
        if self.remove_horizontal_rule == "none":
            mask = np.ones((ny, nx-1), dtype=bool)
        else:
            ii = np.arange(nx-1)[None, :]
            jj = np.arange(ny)[:, None]
            # horizontal element between i and i+1 inherits parity of left node (i,j)
            parity = (ii + jj) % 2  # 0 if even, 1 if odd
            mask = (parity != 0)

        # top rows and bottom rows should be horizontally connected respectively
        if self.keep_boundary_horiz and ny > 0:
            mask[0, :] = True
            mask[-1, :] = True
        return mask

    def _cap_half_size_np(self):
        s = 0.5 * self._thickness0
        return float(np.clip(s, 1e-12, 0.5 * self._thickness0))

    def _compute_node_centers(self):
        nx, ny = self.grid_size
        px, py = self.pitch
        i_idx = np.arange(nx)[None, :].repeat(ny, axis=0)
        j_idx = np.arange(ny)[:, None].repeat(nx, axis=1)
        sign = np.where(((i_idx + j_idx) % 2) == 1, -1.0, 1.0)
        cx = i_idx * px + sign * self.x_offset
        cy = j_idx * py
        return cx, cy

    def _compute_corners_from_params(self):
        cx, cy = self._compute_node_centers()
        s = self._cap_half_size()
        TL = np.stack([cx - s, cy + s], axis=-1)
        TR = np.stack([cx + s, cy + s], axis=-1)
        BR = np.stack([cx + s, cy - s], axis=-1)
        BL = np.stack([cx - s, cy - s], axis=-1)
        return TL, TR, BR, BL

    # ----------------------------
    # Mesh generation (vectorized)
    def _generate_mesh_vectorized(self):
        nx, ny = self.grid_size
        px, py = self._pitch0      # use frozen copies
        q = self.quantize

        # Node centers P[j,i]
        i_idx = np.arange(nx)[None, :].repeat(ny, axis=0)
        j_idx = np.arange(ny)[:, None].repeat(nx, axis=1)
        sign = np.where(((i_idx + j_idx) % 2) == 1, -1.0, 1.0)
        cx = i_idx * px + sign * self._x_offset0
        cy = j_idx * py

        # Corner half size
        s_eff = self._cap_half_size_np()
        include_caps = True

        # Corner arrays (ny,nx,2)
        TL = np.stack([cx - s_eff, cy + s_eff], axis=-1)
        TR = np.stack([cx + s_eff, cy + s_eff], axis=-1)
        BR = np.stack([cx + s_eff, cy - s_eff], axis=-1)
        BL = np.stack([cx - s_eff, cy - s_eff], axis=-1)

        # Store joints as arrays for downstream access
        self.joints = {"TL": TL, "TR": TR, "BR": BR, "BL": BL}

        quads = []

        # (A) Optional joint caps as quads (two tris per cap)
        if include_caps:
            # flatten caps: [TL, TR, BR, BL] for every node
            caps = np.stack([TL, TR, BR, BL], axis=2)  # (ny,nx,4,2)
            quads.append(caps.reshape(-1, 4, 2))

        # (B) Horizontal link quads between (j,i) -- (j,i+1)
        if nx > 1:
            mask_h = self._keep_horizontal_mask(nx, ny)  # (ny, nx-1)
            if mask_h.any():
                # left node sides: R = (TR, BR), right node sides: L = (TL, BL)
                left_TR = TR[:, :-1, :]
                left_BR = BR[:, :-1, :]
                right_TL = TL[:, 1:, :]
                right_BL = BL[:, 1:, :]
                # Quad [left.TR, left.BR, right.BL, right.TL]
                qh = np.stack([left_TR, left_BR, right_BL, right_TL], axis=2)  # (ny, nx-1, 4, 2)
                qh = qh[mask_h]
                if qh.size:
                    quads.append(qh)

        # (C) Vertical link quads between (j,i) -- (j+1,i)
        if ny > 1:
            # Quad [lower.TR, lower.TL, upper.BL, upper.BR]
            lower_TR = TR[:-1, :, :]
            lower_TL = TL[:-1, :, :]
            upper_BL = BL[1:, :, :]
            upper_BR = BR[1:, :, :]
            qv = np.stack([lower_TR, lower_TL, upper_BL, upper_BR], axis=2)  # (ny-1, nx, 4, 2)
            quads.append(qv.reshape(-1, 4, 2))

        # (D) Diagonals (optional)
        if self.add_diagonals and nx > 1 and ny > 1:
            # NE diagonals: between (j,i) and (j+1,i+1)
            a_TR = TR[:-1, :-1, :]
            a_TR2 = TR[:-1, :-1, :]  # same point for consistent winding
            b_TL = TL[1:, 1:, :]
            b_BL = BL[1:, 1:, :]
            q_ne = np.stack([a_TR, a_TR2, b_TL, b_BL], axis=2).reshape(-1, 4, 2)
            quads.append(q_ne)

            # NW diagonals: between (j,i) and (j+1,i-1)
            a_TL = TL[:-1, 1:, :]
            a_TL2 = TL[:-1, 1:, :]
            b_BR = BR[1:, :-1, :]
            b_BR2 = BR[1:, :-1, :]
            q_nw = np.stack([a_TL, a_TL2, b_BR, b_BR2], axis=2).reshape(-1, 4, 2)
            quads.append(q_nw)

        if not quads:
            # Degenerate case: ensure we produce an empty mesh gracefully
            self._verts_np = np.zeros((0, 2), dtype=np.float64)
            self._tris_np = np.zeros((0, 3), dtype=np.int64)
            return

        quads = np.concatenate(quads, axis=0)  # (Nq,4,2)

        # Triangulate quads consistently: (0,1,2) and (0,2,3)
        triA = quads[:, [0, 1, 2], :]  # (Nq,3,2)
        triB = quads[:, [0, 2, 3], :]  # (Nq,3,2)
        tris_pts = np.concatenate([triA, triB], axis=0).reshape(-1, 2)  # (2*Nq*3, 2)

        # ---- Build parallel corner provenance for each point in tris_pts ----
        nx, ny = self.grid_size
        jgrid = np.arange(ny)[:, None].repeat(nx, axis=1)
        igrid = np.arange(nx)[None, :].repeat(ny, axis=0)

        CORNER = {"TL": 0, "TR": 1, "BR": 2, "BL": 3}
        
        def quad_meta_from_nodes(nodes_four):
            j0, i0, c0 = nodes_four[0]
            j1, i1, c1 = nodes_four[1]
            j2, i2, c2 = nodes_four[2]
            j3, i3, c3 = nodes_four[3]
            return np.stack([
                np.stack([j0, i0, np.full_like(j0, c0)], axis=-1),
                np.stack([j1, i1, np.full_like(j1, c1)], axis=-1),
                np.stack([j2, i2, np.full_like(j2, c2)], axis=-1),
                np.stack([j3, i3, np.full_like(j3, c3)], axis=-1),
            ], axis=2)  # (...,4,3)

        meta_blocks = []
        if include_caps:
            jm, im = jgrid, igrid
            caps_meta = quad_meta_from_nodes([
                (jm, im, CORNER["TL"]),
                (jm, im, CORNER["TR"]),
                (jm, im, CORNER["BR"]),
                (jm, im, CORNER["BL"]),
            ])  # (ny,nx,4,3)
            meta_blocks.append(caps_meta.reshape(-1, 4, 3))

        if nx > 1:
            mask_h = self._keep_horizontal_mask(nx, ny)
            if mask_h.any():
                jm = jgrid[:, :-1]
                imL = igrid[:, :-1]
                imR = igrid[:, 1:]
                qh_meta = quad_meta_from_nodes([
                    (jm, imL, CORNER["TR"]),
                    (jm, imL, CORNER["BR"]),
                    (jm, imR, CORNER["BL"]),
                    (jm, imR, CORNER["TL"]),
                ])
                qh_meta = qh_meta[mask_h]
                if qh_meta.size:
                    meta_blocks.append(qh_meta)

        if ny > 1:
            jmB = jgrid[:-1, :]
            jmT = jgrid[1:, :]
            im  = igrid[:-1, :]
            qv_meta = quad_meta_from_nodes([
                (jmB, im, CORNER["TR"]),
                (jmB, im, CORNER["TL"]),
                (jmT, im, CORNER["BL"]),
                (jmT, im, CORNER["BR"]),
            ])
            meta_blocks.append(qv_meta.reshape(-1, 4, 3))

        if self.add_diagonals and nx > 1 and ny > 1:
            jmA = jgrid[:-1, :-1]; imA = igrid[:-1, :-1]
            jmB2= jgrid[1:,  1: ]; imB2= igrid[1:,  1: ]
            q_ne_meta = quad_meta_from_nodes([
                (jmA, imA, CORNER["TR"]),
                (jmA, imA, CORNER["TR"]),
                (jmB2, imB2, CORNER["TL"]),
                (jmB2, imB2, CORNER["BL"]),
            ])
            meta_blocks.append(q_ne_meta.reshape(-1, 4, 3))

            jmA = jgrid[:-1, 1: ]; imA = igrid[:-1, 1: ]
            jmB2= jgrid[1:,  :-1]; imB2= igrid[1:,  :-1]
            q_nw_meta = quad_meta_from_nodes([
                (jmA, imA, CORNER["TL"]),
                (jmA, imA, CORNER["TL"]),
                (jmB2, imB2, CORNER["BR"]),
                (jmB2, imB2, CORNER["BR"]),
            ])
            meta_blocks.append(q_nw_meta.reshape(-1, 4, 3))

        quads_meta = np.concatenate(meta_blocks, axis=0)  # (Nq,4,3)
        triA_meta  = quads_meta[:, [0, 1, 2], :]
        triB_meta  = quads_meta[:, [0, 2, 3], :]
        tris_meta  = np.concatenate([triA_meta, triB_meta], axis=0).reshape(-1, 3)  # (2*Nq*3,3)

        # ---- Robust dedup on integer grid (portable) ----
        q = self.quantize
        qk = np.round(tris_pts / q).astype(np.int64)  # (M,2)
        # Unique rows with first occurrence index and inverse map
        uniq_grid, first_idx, inverse = np.unique(qk, axis=0, return_index=True, return_inverse=True)
        verts_np = (uniq_grid * q).astype(np.float64)         # (nv,2)
        I        = inverse.reshape(-1, 3).astype(np.int64)    # (nt,3)

        # Cache per-vertex (j,i,corner_id) from first occurrence (NumPy only here)
        self._corner_ref_np = tris_meta[first_idx].astype(np.int64)  # (nv,3) int64

        self._verts_np = verts_np
        self._tris_np  = I

    # ----------------------------
    # Torch conversion + stats
    # ----------------------------
    def _generate_torch_mesh(self):
        device = self.pitch.device  # pitch is a Parameter, so this is reliable

        if self._verts_np.size == 0:
            self.verts_torch = torch.empty((0, 2), dtype=torch.float32, device=device)
            self.tris_torch  = torch.empty((0, 3), dtype=torch.int64,  device=device)
            self.nv = 0
            self.nt = 0
            self._corner_ref = torch.empty((0, 3), dtype=torch.long, device=device)
            return

        # connectivity is fixed: just convert to tensor
        self.tris_torch = torch.from_numpy(self._tris_np.astype(np.int64)).to(device)
        self.nt = int(self.tris_torch.shape[0])

        corner_ref_t = torch.from_numpy(self._corner_ref_np.astype(np.int64)).to(device)
        self._corner_ref = corner_ref_t
        self.nv = int(corner_ref_t.shape[0])

        TL, TR, BR, BL = self._compute_corners_from_params_torch()

        jiC = self._corner_ref  # (nv,3)
        j = jiC[:, 0]
        i = jiC[:, 1]
        c = jiC[:, 2]

        corners_stack = torch.stack([TL, TR, BR, BL], dim=0)  # (4,ny,nx,2)
        self.verts_torch = corners_stack[c, j, i]             # (nv,2)


    def _compute_boundary_ids(self):
        device = self.verts_torch.device
        if self.nv == 0:
            self.bottom_ids = torch.empty(0, dtype=torch.int64, device=device)
            self.top_ids    = torch.empty(0, dtype=torch.int64, device=device)
            return

        y = self.verts_torch[:, 1]
        ymin = torch.min(y)
        ymax = torch.max(y)
        tol = max(self.quantize, 1e-12)
        self.bottom_ids = torch.nonzero(torch.isclose(y, ymin, atol=tol), as_tuple=False).squeeze(1)
        self.top_ids    = torch.nonzero(torch.isclose(y, ymax, atol=tol), as_tuple=False).squeeze(1)

    def _cap_half_size_torch(self):
        th = self.thickness          # 0-D tensor (depends on pitch, x_offset)
        s = 0.5 * th
        eps = torch.as_tensor(1e-12, device=th.device, dtype=th.dtype)
        s = torch.clamp(s, min=eps)
        return s

    def _compute_node_centers_torch(self):
        """
        Returns:
            cx, cy: (ny, nx) torch tensors depending on pitch and x_offset.
        """
        device = self.pitch.device
        px, py = self.pitch  # (2,) tensor

        nx, ny = self.grid_size
        i_idx = torch.arange(nx, device=device).view(1, nx).repeat(ny, 1)  # (ny,nx)
        j_idx = torch.arange(ny, device=device).view(ny, 1).repeat(1, nx)  # (ny,nx)

        parity = (i_idx + j_idx) % 2
        sign = torch.where(parity == 1, torch.full_like(parity, -1.0, dtype=torch.float32),
                                      torch.full_like(parity,  1.0, dtype=torch.float32))

        cx = i_idx.to(torch.float32) * px + sign * self.x_offset
        cy = j_idx.to(torch.float32) * py
        return cx, cy

    def _compute_corners_from_params_torch(self):
        cx, cy = self._compute_node_centers_torch()
        if self.joint_cap != "none":
            s = self._cap_half_size_torch()
        else:
            s = torch.tensor(1e-9, device=cx.device, dtype=cx.dtype)

        TL = torch.stack([cx - s, cy + s], dim=-1)
        TR = torch.stack([cx + s, cy + s], dim=-1)
        BR = torch.stack([cx + s, cy - s], dim=-1)
        BL = torch.stack([cx - s, cy - s], dim=-1)
        return TL, TR, BR, BL

    
    def adjust_geometry(self, pitch_new=None, x_offset_new=None):
        """
        O(nv) geometry update: recompute node corners and move existing vertices.

        Thickness is NOT set here: it is a dependent function of (pitch, x_offset)
        through thickness_from_constraint_torch and is used inside the graph.
        """
        device = self.pitch.device

        # Optional manual overwrite of parameters (no grad) – just changing
        # their *values*, not creating a new graph op.
        with torch.no_grad():
            if pitch_new is not None:
                if isinstance(pitch_new, (int, float)):
                    pitch_new = (float(pitch_new), float(pitch_new))
                elif len(pitch_new) != 2:
                    raise ValueError("pitch_new must be a float or (px, py) tuple")
                self.pitch.copy_(torch.tensor(pitch_new, device=device, dtype=torch.float32))

            if x_offset_new is not None:
                self.x_offset.copy_(torch.tensor(float(x_offset_new), device=device))

        if self.nv == 0:
            return self

        # Differentiable rebuild w.r.t pitch and x_offset (and constrained thickness)
        TL, TR, BR, BL = self._compute_corners_from_params_torch()

        jiC = self._corner_ref  # (nv,3) long
        j = jiC[:, 0]
        i = jiC[:, 1]
        c = jiC[:, 2]

        corners_stack = torch.stack([TL, TR, BR, BL], dim=0)  # (4,ny,nx,2)
        new_verts = corners_stack[c, j, i]                    # (nv,2)

        # verts_torch is now a tensor *in the graph* of pitch/x_offset.
        self.verts_torch = new_verts

        return self