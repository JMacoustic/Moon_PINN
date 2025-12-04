import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiFourierNodeModel(nn.Module):
    """
    Time-continuous displacement field on a fixed mesh (verts).

    For each node i at time t:
        [u_i(t), v_i(t)] = NN( x_i, y_i, γ(t) )

    where γ(t) = [t, sin(ω_k t), cos(ω_k t)]_{k=1..K} with learnable ω_k > 0.

    eval_with_derivs(t) uses autograd to get du/dt, d²u/dt², etc.

    API matches your old model:
      - __init__(train_cfg, mesh, device, width, depth, act)
      - set_mesh(mesh)
      - eval_with_derivs(t) -> (disp, vel, acc) shaped as (*t.shape, N_verts, 2)
      - forward(t) -> disp
    """

    def __init__(
        self,
        train_cfg,
        mesh,
        device,
        width: int = 64,
        depth: int = 2,
        act=nn.ELU,
    ):
        super().__init__()

        self.device = device
        self.mesh = mesh

        # ---- time Fourier config ----
        self.K = int(train_cfg.fourier_K)
        assert self.K >= 1, "Need at least K >= 1 for time Fourier features."

        # base physical driving frequency (used for top BC only)
        self.omega_base = 2.0 * math.pi * float(train_cfg.f_top)

        # learnable time frequencies ω_k > 0 via softplus
        init_omega = torch.linspace(1.0, float(self.K), self.K) * self.omega_base
        self.omega_raw = nn.Parameter(init_omega)  # (K,)

        # mesh vertices (keeps grad path from geometry → verts)
        verts = mesh.verts_torch.to(device)
        self.register_buffer("verts", verts)
        self.N_verts = verts.shape[0]

        # optional BC indices as buffers
        if hasattr(mesh, "top_ids") and mesh.top_ids is not None and len(mesh.top_ids) > 0:
            top_idx = torch.as_tensor(mesh.top_ids, device=device, dtype=torch.long)
            self.register_buffer("top_ids", top_idx)
        else:
            self.top_ids = None

        if hasattr(mesh, "bottom_ids") and mesh.bottom_ids is not None and len(mesh.bottom_ids) > 0:
            bottom_idx = torch.as_tensor(mesh.bottom_ids, device=device, dtype=torch.long)
            self.register_buffer("bottom_ids", bottom_idx)
        else:
            self.bottom_ids = None

        # top boundary amplitude v_top(t) = V0 sin(ω_base t)
        self.V0 = float(getattr(train_cfg, "V0", 0.0))

        # ---- MLP trunk ----
        # input per sample: [x, y] + [t, sin(ω_k t), cos(ω_k t)]
        t_feat_dim = 1 + 2 * self.K
        in_dim = 2 + t_feat_dim

        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.trunk = nn.Sequential(*layers)

        # output: [u, v]
        self.head = nn.Linear(width, 2)

        self._init_weights()

    # ---------------- init / mesh update ----------------

    def _init_weights(self):
        for m in self.trunk:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_mesh(self, mesh):
        """
        Keep your existing API: update mesh reference and BC ids.
        """
        self.mesh = mesh
        verts = mesh.verts_torch.to(self.device)
        self.verts = verts
        self.N_verts = verts.shape[0]

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

    # ---------------- time Fourier features ----------------

    def _time_features(self, t_flat: torch.Tensor) -> torch.Tensor:
        """
        t_flat: (B, 1) → (B, 1 + 2K) = [t, sin(ω_k t), cos(ω_k t)]
        """
        t_flat = t_flat.to(self.device)
        omega = F.softplus(self.omega_raw).to(self.device)  # (K,) > 0
        wt = t_flat * omega.view(1, -1)                     # (B, K)
        sin_wt = torch.sin(wt)
        cos_wt = torch.cos(wt)
        return torch.cat([t_flat, sin_wt, cos_wt], dim=-1)  # (B, 1+2K)

    # ---------------- core per-sample forward ----------------

    def _uv_pointwise(self, verts_rep: torch.Tensor, t_rep: torch.Tensor) -> torch.Tensor:
        """
        verts_rep: (B, 2)
        t_rep:     (B, 1)
        returns:   (B, 2) = [u, v]
        """
        t_feat = self._time_features(t_rep)                 # (B, 1+2K)
        inp = torch.cat([verts_rep, t_feat], dim=-1)        # (B, in_dim)
        h = self.trunk(inp)
        uv = self.head(h)                                   # (B, 2)
        return uv

    # ---------------- coeff API (not supported) ----------------

    def _coeffs(self):
        """
        Old code that relied on Fourier coefficients must not silently work.
        """
        raise NotImplementedError(
            "_coeffs() is not defined for the time-Fourier-feature MLP. "
            "Use time-sampled quantities (e.g., loss_bottom_vibration_tsample)."
        )

    # ---------------- eval_with_derivs (autograd in time) ----------------
    def _disp_field_raw(self, t: torch.Tensor):
        """
        Raw displacement field from the network (no IC enforcement, no BC).

        t: (N_t,) or any shape
        returns:
            disp_raw: (N_t, N_verts, 2)
            t_flat:   (N_t, 1)
        """
        device = self.device
        t = torch.as_tensor(t, device=device, dtype=self.verts.dtype)
        t_flat = t.reshape(-1, 1)             # (N_t,1)
        N_t = t_flat.shape[0]
        N_v = self.N_verts

        verts_rep = self.verts.unsqueeze(0).expand(N_t, N_v, 2).reshape(-1, 2)
        t_rep     = t_flat.unsqueeze(1).expand(N_t, N_v, 1).reshape(-1, 1)

        uv = self._uv_pointwise(verts_rep, t_rep)  # (N_t*N_v, 2)
        u = uv[:, 0].view(N_t, N_v, 1)
        v = uv[:, 1].view(N_t, N_v, 1)
        disp_raw = torch.cat([u, v], dim=-1)       # (N_t, N_v, 2)
        return disp_raw, t_flat

    def _disp_field(self, t: torch.Tensor):
        """
        IC-enforced displacement field.

        disp(x,t) = disp_raw(x,t) - disp_raw(x,0)
        """
        # raw at queried times
        disp_raw, t_flat = self._disp_field_raw(t)      # (N_t, N_verts, 2)

        # raw at t = 0 (single time)
        t0 = torch.zeros(1, device=self.device, dtype=self.verts.dtype)
        disp0_raw, _ = self._disp_field_raw(t0)         # (1, N_verts, 2)
        disp0 = disp0_raw[0]                            # (N_verts, 2)

        # subtract initial field (broadcast over time)
        disp = disp_raw - disp0.unsqueeze(0)            # (N_t, N_verts, 2)
        return disp, t_flat


    def eval_with_derivs(self, t: torch.Tensor):
        """
        t: tensor of arbitrary shape

        returns:
            disp, vel, acc with shape (*t.shape, N_verts, 2)

        Uses finite differences in time, but evaluates the network only once
        on the packed time grid [t-h, t, t+h, 0].
        """
        device = self.device
        t = torch.as_tensor(t, device=device, dtype=self.verts.dtype)
        orig_shape = t.shape
        t_flat = t.reshape(-1)                  # (N_t,)
        N_t = t_flat.shape[0]
        N_v = self.N_verts

        # FD step size
        f_top = float(self.omega_base / (2.0 * math.pi))
        T_base = 1.0 / max(f_top, 1e-3)

        h = getattr(self, "dt_fd", None)
        if h is None:
            h = T_base / 50.0
            self.dt_fd = h

        t_minus = t_flat - h                    # (N_t,)
        t_plus  = t_flat + h                    # (N_t,)
        t_zero  = torch.zeros(1, device=device, dtype=self.verts.dtype)  # (1,)

        # Pack all times: [t-h, t, t+h, 0]
        t_all = torch.cat([t_minus, t_flat, t_plus, t_zero], dim=0)      # (3N_t+1,)

        # One big forward pass (raw field, no IC / BC)
        disp_raw_all, _ = self._disp_field_raw(t_all)  # (3N_t+1, N_v, 2)

        disp_m_raw = disp_raw_all[0:N_t]               # at t-h
        disp0_raw  = disp_raw_all[N_t:2 * N_t]         # at t
        disp_p_raw = disp_raw_all[2 * N_t:3 * N_t]     # at t+h
        disp0_raw0 = disp_raw_all[-1]                  # (N_v,2), at t=0

        # IC: u(x,t) = u_raw(x,t) - u_raw(x,0)
        disp = disp0_raw - disp0_raw0.unsqueeze(0)     # (N_t, N_v, 2)

        # FD derivatives on raw field (IC cancels out anyway)
        vel = (disp_p_raw - disp_m_raw) / (2.0 * h)    # (N_t, N_v, 2)
        acc = (disp_p_raw - 2.0 * disp0_raw + disp_m_raw) / (h * h)

        # ---- apply BCs in physical space ----
        if self.bottom_ids is not None and len(self.bottom_ids) > 0:
            bidx = self.bottom_ids.to(device)
            disp[:, bidx, 0] = 0.0
            vel[:,  bidx, 0] = 0.0
            acc[:,  bidx, 0] = 0.0

        if self.top_ids is not None and len(self.top_ids) > 0:
            tidx = self.top_ids.to(device)

            # fixed horizontal
            disp[:, tidx, 0] = 0.0
            vel[:,  tidx, 0] = 0.0
            acc[:,  tidx, 0] = 0.0

            if self.V0 != 0.0:
                t_grid = t_flat.view(N_t, 1)           # (N_t,1)
                w = self.omega_base
                s = torch.sin(w * t_grid)
                c = torch.cos(w * t_grid)

                disp[:, tidx, 1] = self.V0 * s
                vel[:,  tidx, 1] = self.V0 * w * c
                acc[:,  tidx, 1] = -self.V0 * (w * w) * s

        out_shape = (*orig_shape, N_v, 2)
        disp = disp.view(out_shape)
        vel  = vel.view(out_shape)
        acc  = acc.view(out_shape)
        return disp, vel, acc

    def forward(self, t: torch.Tensor):
        disp, _, _ = self.eval_with_derivs(t)
        return disp
