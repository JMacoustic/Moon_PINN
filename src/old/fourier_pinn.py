import math
import torch
import torch.nn as nn
from torch import autograd

class Nondim:
    def __init__(self, L0, U0, E0, rho0):
        self.L0  = float(L0)
        self.U0  = float(U0)
        self.E0  = float(E0)
        self.rho = float(rho0)
        # wave speed and time/stress scales
        c        = (E0 / rho0) ** 0.5   # choose E0 or mu for c; be consistent
        self.T0  = self.L0 / c
        self.S0  = self.E0 * (self.U0 / self.L0)      # stress
        self.D0  = self.S0 / self.L0                  # divergence scale

    # helpers
    def scale_disp(self, q):    # u, v
        return q / self.U0

    def scale_vel(self, qt):    # du/dt
        return qt * self.T0 / self.U0

    def scale_acc(self, qtt):   # d2u/dt2
        return qtt * (self.T0**2) / self.U0

    def scale_stress(self, s):  # sigma
        return s / self.S0

    def scale_div(self, d):     # divergence of sigma or rx, ry
        return d / self.D0

    def mb_hat(self, m_b):
        # dimensionless lumped mass parameter
        return (m_b * self.L0) / (self.E0 * (self.T0**2))  # ~ m_b / (rho * L0)



# -------------- Elasticity utilities --------------
class Elastic2D:
    def __init__(self, mat):
        E, nu = float(mat.E), float(mat.nu)
        if mat.plane_stress:
            # Plane stress
            self.lmbda = E * nu / (1.0 - nu**2)
            self.mu    = E / (2.0 * (1.0 + nu))
        else:
            # Plane strain
            self.lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.mu    = E / (2.0 * (1.0 + nu))
        self.rho = float(mat.rho)

    @torch.no_grad()
    def __repr__(self):
        return f"Elastic2D(lambda={self.lmbda:.3g}, mu={self.mu:.3g}, rho={self.rho:.3g})"

    def stress(self, dux, duy, dvx, dvy):
        """Return σxx, σyy, σxy (small strain, isotropic)."""
        exx = dux
        eyy = dvy
        exy = 0.5 * (duy + dvx)
        lam, mu = self.lmbda, self.mu
        tr = exx + eyy
        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy
        return sxx, syy, sxy


# -------------- Model --------------
# -------------- Fourier-output model --------------
class FourierMLP(nn.Module):
    """
    Predicts spatially varying Fourier coefficients for u,v.
    u(x,y,t) = a0_u + sum_{k=1..K}[ a_k_u cos(k*ω t) + b_k_u sin(k*ω t) ]
    v(x,y,t) = a0_v + sum_{k=1..K}[ a_k_v cos(k*ω t) + b_k_v sin(k*ω t) ]
    """
    def __init__(self, f0: float, K: int = 3, in_dim_spatial: int = 2,
                 width: int = 128, depth: int = 6, act=nn.SiLU):
        super().__init__()
        self.K = int(K)
        self.omega = 2.0 * math.pi * float(f0)

        # trunk over (x,y)
        layers = [nn.Linear(in_dim_spatial, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        self.trunk = nn.Sequential(*layers)

        # heads for coefficients: per component we need (2K+1) coeffs
        ncoef = 2 * self.K + 1
        self.head_u = nn.Linear(width, ncoef)  # [a0_u, a1_u, b1_u, a2_u, b2_u, ...]
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

    def forward(self, x, y, t):
        """
        x, y, t: 1D tensors (broadcastable) with same shape after broadcast.
        Returns u, v with same shape.
        """
        # Ensure shapes align for broadcasting
        # Flatten to 1D for trunk eval, then reshape back
        xy = torch.stack([x, y], dim=-1)
        shp = xy.shape[:-1]  # original leading shape
        xy_flat = xy.view(-1, 2)

        feat = self.trunk(xy_flat)           # (N, width)
        coef_u = self.head_u(feat)           # (N, 2K+1)
        coef_v = self.head_v(feat)           # (N, 2K+1)

        coef_u = coef_u.view(*shp, -1)
        coef_v = coef_v.view(*shp, -1)

        # Build Fourier series at time t
        # layout: [a0, a1, b1, a2, b2, ... , aK, bK]
        a0_u = coef_u[..., 0]
        a0_v = coef_v[..., 0]

        if self.K == 0:
            u = a0_u
            v = a0_v
            return u, v

        # time terms
        wt = self.omega * t
        u_val = a0_u
        v_val = a0_v
        # Accumulate harmonics
        for k in range(1, self.K + 1):
            a_k_u = coef_u[..., 2*k - 1]
            b_k_u = coef_u[..., 2*k]
            a_k_v = coef_v[..., 2*k - 1]
            b_k_v = coef_v[..., 2*k]

            u_val = u_val + a_k_u * torch.cos(k * wt) + b_k_u * torch.sin(k * wt)
            v_val = v_val + a_k_v * torch.cos(k * wt) + b_k_v * torch.sin(k * wt)

        return u_val, v_val



# -------------- PINN --------------
class ElastodynamicsPINN(nn.Module):
    def __init__(self, mesh, mat, cfg):
        super().__init__()
        self.mesh = mesh
        self.mat  = Elastic2D(mat)
        self.cfg  = cfg

        # --- swap to Fourier model (uses cfg.f_top and optional cfg.fourier_K) ---
        K = int(getattr(cfg, "fourier_K", 3))
        self.model = FourierMLP(f0=cfg.f_top, K=K, in_dim_spatial=2, width=128, depth=6, act=nn.SiLU)
        # -------------------------------------------------------------------------

        # ----- Non-dimensionalize -----
        L0  = self.mesh.pitch[1]
        U0  = cfg.V0
        E0  = mat.E
        rho0= self.mat.rho
        self.nd = Nondim(L0, U0, E0, rho0)

        # --- BC params ---
        self.bottom_mode = getattr(cfg, "bottom_mode", "mass")
        self.payload_P0 = float(getattr(cfg, "payload_P0", 0.0))
        self.m_bottom = float(getattr(cfg, "m_bottom", 0.0))
        self.mass_node_idx = None

    # Optional: notify the model when the mesh geometry changes
    def set_mesh(self, mesh):
        self.mesh = mesh
    
    def set_bottom_mass(self, mass: float, node_index: int | None = None):
        """Assign a lumped mass to one bottom node (free node)."""
        self.m_bottom = float(mass)
        if self.mesh.bottom_ids.numel() == 0:
            self.mass_node_idx = None
            return
        if node_index is not None:
            self.mass_node_idx = int(node_index)
            return
        # default: node closest to bottom-edge centroid in x
        B = self.mesh.verts_torch[self.mesh.bottom_ids].to(next(self.parameters()).device)
        xB = B[:, 0]
        x0 = xB.mean()
        self.mass_node_idx = int((xB - x0).abs().argmin().item())

    # ----- core -----
    def forward(self, x, y, t):
        # call the Fourier model (same signature for the PINN)
        return self.model(x, y, t)

    def _first_derivs(self, u, v, x, y, t):
        """Compute du/dx, du/dy, du/dt and dv/dx, dv/dy, dv/dt with 2 autograd calls."""
        ones_u = torch.ones_like(u)
        ones_v = torch.ones_like(v)
        dux, duy, dut = autograd.grad(
            u, (x, y, t), grad_outputs=ones_u, create_graph=True, retain_graph=True
        )
        dvx, dvy, dvt = autograd.grad(
            v, (x, y, t), grad_outputs=ones_v, create_graph=True, retain_graph=True
        )
        return dux, duy, dut, dvx, dvy, dvt

    def pde_residual(self, x, y, t):
        x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
        u, v = self.forward(x, y, t)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(u, v, x, y, t)

        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)
        dsxx_dx = autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True, retain_graph=True)[0]
        dsxy_dy = autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]
        dsyy_dy = autograd.grad(syy, y, grad_outputs=torch.ones_like(syy), create_graph=True, retain_graph=True)[0]
        dsxy_dx = autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]

        du_tt = autograd.grad(dut, t, grad_outputs=torch.ones_like(dut), create_graph=True, retain_graph=True)[0]
        dv_tt = autograd.grad(dvt, t, grad_outputs=torch.ones_like(dvt), create_graph=True, retain_graph=True)[0]

        rx = self.mat.rho * du_tt - (dsxx_dx + dsxy_dy)
        ry = self.mat.rho * dv_tt - (dsxy_dx + dsyy_dy)

        # ---- make unitless ----
        rx_hat = self.nd.scale_div(rx)
        ry_hat = self.nd.scale_div(ry)
        return rx_hat, ry_hat


    # ----- boundary/initial conditions -----
    def bc_bottom(self, time_steps: int = 25):
        if self.bottom_mode == "clamp":
            return self.bc_bottom_clamp(time_steps=time_steps)
        if self.bottom_mode == "mass":
            return self.bc_bottom_mass()
        if self.bottom_mode == "payload":
            return self.bc_bottom_payload(time_steps=time_steps)

    def bc_bottom_clamp(self, time_steps: int = 25):
        """
        OPTION 3: Clamp the entire bottom edge (Dirichlet).
        Enforce u = 0 and/or v = 0 (dimensionless) over time on all bottom nodes.
        """
        device = next(self.parameters()).device
        if self.mesh.bottom_ids.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        # bottom nodes (Nb, 2)
        B  = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)
        xB, yB = B[:, 0], B[:, 1]
        Nb = xB.numel()

        # time grid
        T = time_steps
        times = torch.linspace(0.0, self.cfg.T, steps=T, device=device)

        # broadcast (T*Nb,)
        tB   = times[:, None].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)
        xrep = xB[None, :].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)
        yrep = yB[None, :].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)

        # evaluate model
        uB, vB = self.forward(xrep, yrep, tB)

        # dimensionless displacements
        uh = self.nd.scale_disp(uB)   # u / U0
        vh = self.nd.scale_disp(vB)   # v / U0

        comps = getattr(self.cfg, "bottom_clamp_components", "uv")
        if comps == "u":
            return (uh**2).mean()
        elif comps == "v":
            return (vh**2).mean()
        else:  # "uv"
            return 0.5 * ((uh**2).mean() + (vh**2).mean())

    
    def bc_bottom_payload(self, time_steps: int = 25):
        device = next(self.parameters()).device
        if self.mesh.bottom_ids.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        # bottom nodes (Nb, 2)
        B  = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)
        xB, yB = B[:, 0], B[:, 1]
        Nb = xB.numel()

        # time grid
        T = time_steps
        times = torch.linspace(0.0, self.cfg.T, steps=T, device=device)

        # broadcast to (T*Nb,)
        tB   = times[:, None].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)
        xrep = xB[None, :].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)
        yrep = yB[None, :].expand(T, Nb).reshape(-1).contiguous().requires_grad_(True)

        # forward + grads
        uB, vB = self.forward(xrep, yrep, tB)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(uB, vB, xrep, yrep, tB)
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)

        # constant traction
        p_t = torch.full_like(tB, fill_value=self.payload_P0)

        # nondimensional residuals on the boundary
        shear_free_hat  = -self.nd.scale_stress(sxy)                         # -> 0
        normal_match_hat = -self.nd.scale_stress(syy) - (p_t / self.nd.S0)   # -> 0

        return (shear_free_hat**2).mean() + (normal_match_hat**2).mean()

    
    def bc_bottom_mass(self, time_steps: int = 25):
        device = next(self.parameters()).device
        if self.m_bottom <= 0.0 or self.mesh.bottom_ids.numel() == 0:
            return torch.zeros((), device=device)

        if self.mass_node_idx is None or not (0 <= self.mass_node_idx < self.mesh.bottom_ids.numel()):
            self.set_bottom_mass(self.m_bottom, None)
            if self.mass_node_idx is None:
                return torch.zeros((), device=device)

        B = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)
        x0, y0 = B[self.mass_node_idx, 0], B[self.mass_node_idx, 1]

        T = time_steps
        t = torch.linspace(0.0, self.cfg.T, steps=T, device=device).requires_grad_(True)
        x = x0.expand(T).clone().requires_grad_(True)
        y = y0.expand(T).clone().requires_grad_(True)

        u, v = self.forward(x, y, t)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(u, v, x, y, t)
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)
        v_tt = autograd.grad(dvt, t, grad_outputs=torch.ones_like(dvt), create_graph=True, retain_graph=True)[0]

        # dimensionless
        shear_free_hat  = -self.nd.scale_stress(sxy)
        dyn_balance_hat = -self.nd.scale_stress(syy) - self.nd.mb_hat(self.m_bottom) * self.nd.scale_acc(v_tt)

        return (shear_free_hat**2).mean() + (dyn_balance_hat**2).mean()


    def bc_top_disp(self, x, y, t):
        u, v = self.forward(x, y, t)
        v_tgt = self.cfg.V0 * torch.sin(2.0 * math.pi * self.cfg.f_top * t)
        return self.nd.scale_disp(u), self.nd.scale_disp(v - v_tgt)

    def ic_zero(self, x, y, t0):
        t0 = t0.requires_grad_(True)
        u, v = self.forward(x, y, t0)
        du_t = autograd.grad(u, t0, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv_t = autograd.grad(v, t0, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return self.nd.scale_disp(u), self.nd.scale_disp(v), self.nd.scale_vel(du_t), self.nd.scale_vel(dv_t)

    # ----- vibration metric (vectorized over time) -----
    def bottom_vibration_loss(self, time_steps: int = 25, component: str = "v"):
        """
        Dimensionless time-RMS (DC-removed) of bottom-edge motion.
        Returns a scalar. component in {"u","v","uv"}.
        """
        device = next(self.parameters()).device

        # no bottom nodes -> zero loss
        if self.mesh.bottom_ids.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=device)

        # bottom node coords (Nb, 2)
        B  = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)
        xB, yB = B[:, 0], B[:, 1]
        Nb = xB.numel()

        # time grid (T,)
        times = torch.linspace(0.0, self.cfg.T, steps=time_steps, device=device)
        T = times.shape[0]

        # broadcast to (T*Nb,)
        tB   = times[:, None].expand(T, Nb).reshape(-1).contiguous()
        xrep = xB[None, :].expand(T, Nb).reshape(-1)
        yrep = yB[None, :].expand(T, Nb).reshape(-1)

        # network eval -> (T*Nb,) then reshape to (T, Nb)
        uB, vB = self.forward(xrep, yrep, tB)
        U = uB.view(T, Nb)
        V = vB.view(T, Nb)

        # remove DC per node over time
        Uc = U - U.mean(dim=0, keepdim=True)
        Vc = V - V.mean(dim=0, keepdim=True)

        # nondimensionalize displacements by U0
        Uh = self.nd.scale_disp(Uc)  # = Uc / U0
        Vh = self.nd.scale_disp(Vc)  # = Vc / U0

        if component == "u":
            return (Uh**2).mean()
        elif component == "v":
            return (Vh**2).mean()
        elif component == "uv":
            return 0.5 * ((Uh**2).mean() + (Vh**2).mean())
        else:
            # default to vertical if unknown component
            return (Vh**2).mean()
