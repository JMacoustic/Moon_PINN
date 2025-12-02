import math
import torch
import torch.nn as nn

class MultiFourierNodeModel(nn.Module):
    """
    Time-continuous displacement field on a fixed mesh (verts).

    For each vert i:
      u_i(t) = a0_u(i) + sum_{k=1..K}[ a_k_u(i) cos(w_k t) + b_k_u(i) sin(w_k t) ]
      v_i(t) = a0_v(i) + sum_{k=1..K}[ a_k_v(i) cos(w_k t) + b_k_v(i) sin(w_k t) ]

    w_1 = w_base (fixed)
    w_k = freq_scale[k-2] * w_base for k = 2..K (learnable)

    Layout per node: [a0, a1, b1, a2, b2, ..., aK, bK]
    """

    def __init__(
        self,
        train_cfg,
        mesh,
        device,
        width: int = 256,
        depth: int = 4,
        act=nn.ELU,
    ):
        super().__init__()
        self.K = int(train_cfg.fourier_K)
        assert self.K >= 1, "Need at least K >= 1 for fixed BC mode."

        self.omega = 2.0 * math.pi * float(train_cfg.f_top)  # base w
        self.device = device

        # mesh + buffers
        self.mesh = mesh
        verts = mesh.verts_torch.to(device)  # NO .detach()
        self.register_buffer("verts", verts)
        self.N_verts = verts.shape[0]

        if hasattr(mesh, "top_ids") and mesh.top_ids is not None and len(mesh.top_ids) > 0:
            top_idx = torch.as_tensor(mesh.top_ids, device=device, dtype=torch.long)
            self.register_buffer("top_ids", top_idx)
        else:
            self.top_ids = None

        self.V0 = float(getattr(train_cfg, "V0", 0.0))

        # trunk
        layers = [nn.Linear(2, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.trunk = nn.Sequential(*layers)

        # Fourier coefficient heads
        ncoef = 2 * self.K + 1
        self.head_u = nn.Linear(width, ncoef)
        self.head_v = nn.Linear(width, ncoef)

        # learnable frequency multipliers for k=2..K
        if self.K > 1:
            freq_init = torch.arange(2, self.K + 1, dtype=torch.float32)  # [2,3,...,K]
            self.freq_scale = nn.Parameter(freq_init)  # length K-1
        else:
            self.freq_scale = None  # no extra modes

        self._init_weights()

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for head in (self.head_u, self.head_v):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

        # initialize v-mode-1 to V0 sin(w t) roughly (b1 = V0)
        if self.V0 != 0.0:
            with torch.no_grad():
                # layout: [a0, a1, b1, a2, b2, ...]
                self.head_v.bias[2] = self.V0
                
    def set_mesh(self, mesh):
        self.mesh = mesh
        verts = mesh.verts_torch.to(self.device)  # NO .detach()
        self.verts = verts  # keep as buffer; assigning a tensor that has grad_fn is fine

        if hasattr(mesh, "top_ids") and mesh.top_ids is not None and len(mesh.top_ids) > 0:
            top_idx = torch.as_tensor(mesh.top_ids, device=self.device, dtype=torch.long)
            if isinstance(self.top_ids, torch.Tensor):
                self.top_ids = top_idx
            else:
                self.register_buffer("top_ids", top_idx)
        else:
            self.top_ids = None

    def _coeffs(self):
        feat = self.trunk(self.verts)   # (N, width)
        coef_u = self.head_u(feat)      # (N, 2K+1)
        coef_v = self.head_v(feat)

        # enforce zero initial displacement: all cosine terms = 0
        coef_u[:, 0] = 0.0  # a0
        coef_v[:, 0] = 0.0
        for k in range(1, self.K + 1):
            ak_col = 2 * k - 1
            coef_u[:, ak_col] = 0.0
            coef_v[:, ak_col] = 0.0

        # enforce top BC with fixed first mode
        if self.top_ids is not None:
            idx = self.top_ids
            coef_u = coef_u.clone()
            coef_v = coef_v.clone()

            # u ≡ 0
            coef_u[idx, :] = 0.0

            # v = V0 sin(w t) via mode-1
            coef_v[idx, :] = 0.0
            if self.V0 != 0.0:
                coef_v[idx, 2] = self.V0  # b1 for mode-1

        return coef_u, coef_v

    def eval_with_derivs(self, t: torch.Tensor):
        device = self.verts.device
        dtype = self.verts.dtype
        t = torch.as_tensor(t, device=device, dtype=dtype)
        t_flat = t.reshape(-1)  # (B,)
        B = t_flat.shape[0]
        N = self.N_verts
        K = self.K

        coef_u, coef_v = self._coeffs()  # (N, 2K+1)

        a0_u = coef_u[:, 0]
        a0_v = coef_v[:, 0]

        u   = a0_u.unsqueeze(0).expand(B, N).clone()
        v   = a0_v.unsqueeze(0).expand(B, N).clone()
        ut  = torch.zeros_like(u)
        vt  = torch.zeros_like(v)
        utt = torch.zeros_like(u)
        vtt = torch.zeros_like(v)

        if K > 0:
            tB = t_flat.view(B, 1)

            # --- mode 1: fixed w_1 = w_base ---
            w1 = self.omega
            cos_wt = torch.cos(w1 * tB)
            sin_wt = torch.sin(w1 * tB)

            a1_u = coef_u[:, 1]  # a1
            b1_u = coef_u[:, 2]  # b1
            a1_v = coef_v[:, 1]
            b1_v = coef_v[:, 2]

            a1_uB = a1_u.unsqueeze(0)
            b1_uB = b1_u.unsqueeze(0)
            a1_vB = a1_v.unsqueeze(0)
            b1_vB = b1_v.unsqueeze(0)

            u   = u   + a1_uB * cos_wt + b1_uB * sin_wt
            v   = v   + a1_vB * cos_wt + b1_vB * sin_wt

            ut  = ut  + (-a1_uB * w1 * sin_wt + b1_uB * w1 * cos_wt)
            vt  = vt  + (-a1_vB * w1 * sin_wt + b1_vB * w1 * cos_wt)

            w1_2 = w1 * w1
            utt = utt + (-a1_uB * w1_2 * cos_wt - b1_uB * w1_2 * sin_wt)
            vtt = vtt + (-a1_vB * w1_2 * cos_wt - b1_vB * w1_2 * sin_wt)

            # --- modes 2..K: learnable frequencies ---
            if K > 1:
                freq_scale = self.freq_scale.to(device=device, dtype=dtype)  # (K-1,)
                for k in range(2, K + 1):
                    s_k = freq_scale[k - 2]     # α_{k-1}
                    w_k = s_k * self.omega
                    w2  = w_k * w_k

                    cos_wt = torch.cos(w_k * tB)
                    sin_wt = torch.sin(w_k * tB)

                    a_k_u = coef_u[:, 2 * k - 1]
                    b_k_u = coef_u[:, 2 * k]
                    a_k_v = coef_v[:, 2 * k - 1]
                    b_k_v = coef_v[:, 2 * k]

                    a_k_uB = a_k_u.unsqueeze(0)
                    b_k_uB = b_k_u.unsqueeze(0)
                    a_k_vB = a_k_v.unsqueeze(0)
                    b_k_vB = b_k_v.unsqueeze(0)

                    u   = u   + a_k_uB * cos_wt + b_k_uB * sin_wt
                    v   = v   + a_k_vB * cos_wt + b_k_vB * sin_wt

                    ut  = ut  + (-a_k_uB * w_k * sin_wt + b_k_uB * w_k * cos_wt)
                    vt  = vt  + (-a_k_vB * w_k * sin_wt + b_k_vB * w_k * cos_wt)

                    utt = utt + (-a_k_uB * w2 * cos_wt - b_k_uB * w2 * sin_wt)
                    vtt = vtt + (-a_k_vB * w2 * cos_wt - b_k_vB * w2 * sin_wt)

        disp = torch.stack([u, v], dim=-1)
        vel  = torch.stack([ut, vt], dim=-1)
        acc  = torch.stack([utt, vtt], dim=-1)

        out_shape = (*t.shape, N, 2)
        disp = disp.view(out_shape)
        vel  = vel.view(out_shape)
        acc  = acc.view(out_shape)
        return disp, vel, acc

    def forward(self, t: torch.Tensor):
        disp, _, _ = self.eval_with_derivs(t)
        return disp
