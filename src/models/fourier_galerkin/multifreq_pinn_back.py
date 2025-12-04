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
        width: int = 512,
        depth: int = 5,
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

        # if hasattr(mesh, "bottom_ids") and mesh.bottom_ids is not None and len(mesh.bottom_ids) > 0:
        #     bottom_idx = torch.as_tensor(mesh.bottom_ids, device=device, dtype=torch.long)
        #     self.register_buffer("bottom_ids", bottom_idx)
        # else:
        #     self.bottom_ids = None

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

        base = torch.empty(self.K - 1, dtype=torch.float32)
        base[0] = 0.5
        if self.K > 2:
            base[1:] = torch.arange(2, self.K, dtype=torch.float32)  # 2..K
            self.register_buffer("freq_base", base)  # non-trainable integer harmonics

            # learn small perturbations around base, unconstrained
            self.freq_raw = nn.Parameter(torch.zeros(self.K - 1))  # starts at 0 (no shift)
            self.freq_perturb_weight = 0.5
        else:
            self.freq_base = None
            self.freq_raw = None

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
                self.head_v.bias[2] = self.V0 * 4.0 * math.pi / self.omega
                
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
        
        if hasattr(mesh, "bottom_ids") and mesh.bottom_ids is not None and len(mesh.bottom_ids) > 0:
            bottom_idx = torch.as_tensor(mesh.bottom_ids, device=self.device, dtype=torch.long)
            if isinstance(self.bottom_ids, torch.Tensor):
                self.bottom_ids = bottom_idx
            else:
                self.register_buffer("bottom_ids", bottom_idx)
        else:
            self.bottom_ids = None

    def _coeffs(self):
        feat = self.trunk(self.verts)   # (N, width)
        coef_u = self.head_u(feat)
        coef_v = self.head_v(feat)

        # shapes: (N, 1 + 2K)
        a0_u = coef_u[:, 0:1]
        a_u  = coef_u[:, 1:2*self.K:2]  # (N, K) cosine
        b_u  = coef_u[:, 2:2*self.K+1:2]

        a0_v = coef_v[:, 0:1]
        a_v  = coef_v[:, 1:2*self.K:2]
        b_v  = coef_v[:, 2:2*self.K+1:2]

        # enforce u(0)=v(0)=0 (or general IC) hard:
        a0_u = -a_u.sum(dim=1, keepdim=True)  # for u(0)=0
        a0_v = -a_v.sum(dim=1, keepdim=True)  # for v(0)=0

        # now rebuild coef_u/coef_v from (a0, a, b)
        coef_u_new = torch.empty_like(coef_u)
        coef_v_new = torch.empty_like(coef_v)

        coef_u_new[:, 0:1] = a0_u
        coef_u_new[:, 1:2*self.K:2] = a_u
        coef_u_new[:, 2:2*self.K+1:2] = b_u

        coef_v_new[:, 0:1] = a0_v
        coef_v_new[:, 1:2*self.K:2] = a_v
        coef_v_new[:, 2:2*self.K+1:2] = b_v

        coef_u, coef_v = coef_u_new, coef_v_new

        # --- boundary conditions ---
        if (self.top_ids is not None) or (self.bottom_ids is not None):
            coef_u = coef_u.clone()
            coef_v = coef_v.clone()

            # # bottom: horizontal displacement u ≡ 0
            # if self.bottom_ids is not None:
            #     bidx = self.bottom_ids
            #     coef_u[bidx, :] = 0.0 

            # top: u ≡ 0, v = V0 sin(ω t) via first sine mode
            if self.top_ids is not None:
                tidx = self.top_ids
                coef_u[tidx, :] = 0.0
                coef_v[tidx, :] = 0.0 
                if self.V0 != 0.0:
                    coef_v[tidx, 2] = self.V0

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
                # map raw -> bounded perturbation in (-perturb_max, +perturb_max)
                delta = self.freq_perturb_weight * self.freq_raw.to(dtype=dtype, device=device)
                freq_scale = self.freq_base.to(dtype=dtype, device=device) + delta  # shape (K-1,)

                for k in range(2, K + 1):
                    s_k = freq_scale[k - 2]     # ≈ integer + bounded shift
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
