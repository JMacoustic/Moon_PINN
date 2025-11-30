import math
import torch
import torch.nn as nn

class FourierNodeModel(nn.Module):
    """
    Time-continuous displacement field on a fixed mesh (verts).

    For each vert i:
      u_i(t) = a0_u(i) + sum_{k=1..K}[ a_k_u(i) cos(k ω t) + b_k_u(i) sin(k ω t) ]
      v_i(t) = a0_v(i) + sum_{k=1..K}[ a_k_v(i) cos(k ω t) + b_k_v(i) sin(k ω t) ]

    Layout of coeff vector per node: [a0, a1, b1, a2, b2, ..., aK, bK]

    Constraints:
      - Initial displacement: u(0) = 0, v(0) = 0 for all nodes
        → enforced by setting all a-coeffs (a0, a_k) to zero.
      - Top boundary nodes (mesh.top_ids):
        u(t) ≡ 0
        v(t) = V0 sin(ω t) with V0 = train_cfg.V0
    """
    def __init__(
        self,
        train_cfg,
        mesh,
        device,
        width: int = 128,
        depth: int = 4,
        act=nn.SiLU,
    ):
        super().__init__()
        self.K = int(train_cfg.fourier_K)
        self.omega = 2.0 * math.pi * float(train_cfg.f_top)
        self.device = device

        self.mesh = mesh
        verts = mesh.verts_torch.to(device).detach()
        self.register_buffer("verts", verts)
        self.N_verts = verts.shape[0]

        if hasattr(mesh, "top_ids") and mesh.top_ids is not None and len(mesh.top_ids) > 0:
            top_idx = torch.as_tensor(mesh.top_ids, device=device, dtype=torch.long)
            self.register_buffer("top_ids", top_idx)
        else:
            self.top_ids = None

        self.V0 = float(getattr(train_cfg, "V0", 0.0))

        layers = [nn.Linear(2, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.trunk = nn.Sequential(*layers)

        ncoef = 2 * self.K + 1
        self.head_u = nn.Linear(width, ncoef)
        self.head_v = nn.Linear(width, ncoef)

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for head in (self.head_u, self.head_v):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def set_mesh(self, mesh):
        self.mesh = mesh
        verts = mesh.verts_torch.to(self.device).detach()
        self.verts.data = verts

        if hasattr(mesh, "top_ids") and mesh.top_ids is not None and len(mesh.top_ids) > 0:
            top_idx = torch.as_tensor(mesh.top_ids, device=self.device, dtype=torch.long)
            if isinstance(self.top_ids, torch.Tensor):
                self.top_ids.data = top_idx
            else:
                self.register_buffer("top_ids", top_idx)
        else:
            self.top_ids = None

    def _coeffs(self):
        """Return Fourier coeffs for all verts: (N_verts, 2K+1)."""
        feat = self.trunk(self.verts)    # (N, width)
        coef_u = self.head_u(feat)       # (N, 2K+1)
        coef_v = self.head_v(feat)

        # --- enforce zero initial displacement for all nodes ---
        # u(0) = a0_u + sum_k a_k_u = 0
        # v(0) = a0_v + sum_k a_k_v = 0
        # Here we choose the strict version: all a-coeffs are zero.
        if self.K >= 0:
            # a0
            coef_u[:, 0] = 0.0
            coef_v[:, 0] = 0.0
            # a_k for k=1..K
            for k in range(1, self.K + 1):
                ak_col = 2 * k - 1
                coef_u[:, ak_col] = 0.0
                coef_v[:, ak_col] = 0.0

        # --- enforce top boundary sinusoidal BC ---
        if self.top_ids is not None and self.K > 0:
            idx = self.top_ids
            coef_u = coef_u.clone()
            coef_v = coef_v.clone()

            # u ≡ 0 → all coeffs zero
            coef_u[idx, :] = 0.0

            # v = V0 sin(ω t) → a0 = 0, b1 = V0, others = 0
            coef_v[idx, :] = 0.0
            if self.V0 != 0.0:
                # layout: [a0, a1, b1, a2, b2, ...]
                coef_v[idx, 2] = self.V0  # b1

        return coef_u, coef_v

    def forward(self, t: torch.Tensor):
        disp, _, _ = self.eval_with_derivs(t)
        return disp

    def eval_with_derivs(self, t: torch.Tensor):
        """
        Evaluate displacement, velocity, acceleration analytically.

        t: scalar or tensor of shape (...,)

        Returns:
            disp : (*t.shape, N_verts, 2)
            vel  : (*t.shape, N_verts, 2)
            acc  : (*t.shape, N_verts, 2)
        """
        device = self.verts.device
        dtype = self.verts.dtype
        t = torch.as_tensor(t, device=device, dtype=dtype)
        t_flat = t.reshape(-1)           # (B,)
        B = t_flat.shape[0]
        N = self.N_verts
        K = self.K

        coef_u, coef_v = self._coeffs()  # (N, 2K+1)

        a0_u = coef_u[:, 0]
        a0_v = coef_v[:, 0]

        u = a0_u.unsqueeze(0).expand(B, N).clone()
        v = a0_v.unsqueeze(0).expand(B, N).clone()
        ut = torch.zeros_like(u)
        vt = torch.zeros_like(v)
        utt = torch.zeros_like(u)
        vtt = torch.zeros_like(v)

        if K > 0:
            wt_base = t_flat.view(B, 1) * self.omega
            for k in range(1, K + 1):
                a_k_u = coef_u[:, 2 * k - 1]
                b_k_u = coef_u[:, 2 * k]
                a_k_v = coef_v[:, 2 * k - 1]
                b_k_v = coef_v[:, 2 * k]

                kω = k * self.omega
                kω2 = kω * kω

                cos_kwt = torch.cos(k * wt_base)
                sin_kwt = torch.sin(k * wt_base)

                a_k_uB = a_k_u.unsqueeze(0)
                b_k_uB = b_k_u.unsqueeze(0)
                a_k_vB = a_k_v.unsqueeze(0)
                b_k_vB = b_k_v.unsqueeze(0)

                u   = u   + a_k_uB * cos_kwt + b_k_uB * sin_kwt
                v   = v   + a_k_vB * cos_kwt + b_k_vB * sin_kwt

                ut  = ut  + (-a_k_uB * kω * sin_kwt + b_k_uB * kω * cos_kwt)
                vt  = vt  + (-a_k_vB * kω * sin_kwt + b_k_vB * kω * cos_kwt)

                utt = utt + (-a_k_uB * kω2 * cos_kwt - b_k_uB * kω2 * sin_kwt)
                vtt = vtt + (-a_k_vB * kω2 * cos_kwt - b_k_vB * kω2 * sin_kwt)

        disp = torch.stack([u, v], dim=-1)
        vel  = torch.stack([ut, vt], dim=-1)
        acc  = torch.stack([utt, vtt], dim=-1)

        out_shape = (*t.shape, N, 2)
        disp = disp.view(out_shape)
        vel  = vel.view(out_shape)
        acc  = acc.view(out_shape)
        return disp, vel, acc
