import math
import torch
import torch.nn as nn
from torch import autograd

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
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=128, depth=6, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xyt):
        return self.net(xyt)


# -------------- PINN --------------
class ElastodynamicsPINN(nn.Module):
    """
    Expects `mesh` to be your Aux object:
      - mesh.verts_torch (nv,2), mesh.tris_torch (nt,3)
      - mesh.sample_interior(n), mesh.sample_on_nodes(ids,n)
      - mesh.bottom_ids, mesh.top_ids
    """
    def __init__(self, mesh, mat, cfg):
        super().__init__()
        self.mesh = mesh
        self.mat  = Elastic2D(mat)
        self.cfg  = cfg
        self.model = MLP(in_dim=3, out_dim=2, width=128, depth=6, act=nn.SiLU)
        self.m_bottom = float(getattr(cfg, "m_bottom", 0.0))  # lumped mass (per unit thickness)
        self.mass_node_idx = None  # index within bottom_ids

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
        xyt = torch.stack([x, y, t], dim=-1)
        uv = self.model(xyt)
        return uv[..., 0], uv[..., 1]

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
        # enable grads
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)

        u, v = self.forward(x, y, t)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(u, v, x, y, t)

        # stress and its divergence
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)

        dsxx_dx = autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx),
                                create_graph=True, retain_graph=True)[0]
        dsxy_dy = autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy),
                                create_graph=True, retain_graph=True)[0]
        dsyy_dy = autograd.grad(syy, y, grad_outputs=torch.ones_like(syy),
                                create_graph=True, retain_graph=True)[0]
        dsxy_dx = autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy),
                                create_graph=True, retain_graph=True)[0]

        # accelerations
        du_tt = autograd.grad(dut, t, grad_outputs=torch.ones_like(dut),
                              create_graph=True, retain_graph=True)[0]
        dv_tt = autograd.grad(dvt, t, grad_outputs=torch.ones_like(dvt),
                              create_graph=True, retain_graph=True)[0]

        # body forces (0)
        rx = self.mat.rho * du_tt - (dsxx_dx + dsxy_dy)
        ry = self.mat.rho * dv_tt - (dsxy_dx + dsyy_dy)
        return rx, ry

    # ----- boundary/initial conditions -----
    def bc_bottom_clamp(self, x, y, t):
        """u = 0, v = 0 on bottom."""
        u, v = self.forward(x, y, t)
        return u, v
    
    def bc_bottom_mass(self, time_steps: int = 25):
        """
        Boundary only at the selected bottom mass node:
          shear_free_mass : -σ_xy -> 0
          dyn_balance_mass: -σ_yy - m_b * v_tt -> 0
        All other bottom nodes are unconstrained here (free).
        """
        device = next(self.parameters()).device

        # early exits (no BC terms to add)
        if self.m_bottom <= 0.0 or self.mesh.bottom_ids.numel() == 0:
            z = torch.zeros((), dtype=torch.float32, device=device)
            return z, z

        # ensure mass_node_idx is valid
        if self.mass_node_idx is None or not (0 <= self.mass_node_idx < self.mesh.bottom_ids.numel()):
            self.set_bottom_mass(self.m_bottom, None)
            if self.mass_node_idx is None:
                z = torch.zeros((), dtype=torch.float32, device=device)
                return z, z

        # gather the mass node coords (constant in time)
        B = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)  # (Nb,2)
        x0, y0 = B[self.mass_node_idx, 0], B[self.mass_node_idx, 1]

        # time grid
        T = time_steps
        t = torch.linspace(0.0, self.cfg.T, steps=T, device=device).requires_grad_(True)

        # repeat node over time
        x = x0.expand(T).clone().requires_grad_(True)
        y = y0.expand(T).clone().requires_grad_(True)

        # forward & derivs
        u, v = self.forward(x, y, t)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(u, v, x, y, t)

        # stresses at the mass node
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)

        # vertical acceleration v_tt
        v_tt = autograd.grad(dvt, t, grad_outputs=torch.ones_like(dvt),
                             create_graph=True, retain_graph=True)[0]

        # bottom outward normal n=(0,-1): traction t = σ·n -> t_x = -σ_xy, t_y = -σ_yy
        shear_free_mass  = -sxy                        # -> 0
        dyn_balance_mass = (-syy) - self.m_bottom * v_tt  # -> 0

        L = (shear_free_mass**2).mean() + (dyn_balance_mass**2).mean()

        return L

    def bc_top_disp(self, x, y, t):
        """u = 0, v = V0 * 4 * sin(2π f t) on top."""
        u, v = self.forward(x, y, t)
        v_tgt = self.cfg.V0 * 4.0 * torch.sin(2.0 * math.pi * self.cfg.f_top * t)
        return u, v - v_tgt

    def ic_zero(self, x, y, t0):
        """u=v=0, and du/dt=dv/dt=0 at t=0."""
        t0 = t0.requires_grad_(True)
        u, v = self.forward(x, y, t0)
        du_t = autograd.grad(u, t0, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv_t = autograd.grad(v, t0, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        return u, v, du_t, dv_t

    # ----- vibration metric (vectorized over time) -----
    def bottom_vibration_loss(self, time_steps: int = 25, component: str = "v"):
        """
        Mean-square (time-RMS, DC removed) of bottom-edge motion.
        Vectorized over the time dimension for speed.
        """
        if self.mesh.bottom_ids.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        B = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)   # (Nb, 2)
        xB, yB = B[:, 0], B[:, 1]

        # shape (T,), then broadcast to (T, Nb)
        times = torch.linspace(0.0, self.cfg.T, steps=time_steps, device=device)
        tB = times[:, None].expand(-1, xB.numel()).contiguous().view(-1)  # (T*Nb,)
        xB_rep = xB[None, :].expand(times.shape[0], -1).reshape(-1)       # (T*Nb,)
        yB_rep = yB[None, :].expand(times.shape[0], -1).reshape(-1)       # (T*Nb,)

        uB, vB = self.forward(xB_rep, yB_rep, tB)     # (T*Nb,)
        U = uB.view(times.shape[0], -1)               # (T, Nb)
        V = vB.view(times.shape[0], -1)               # (T, Nb)

        # remove DC per node
        Uc = U - U.mean(dim=0, keepdim=True)
        Vc = V - V.mean(dim=0, keepdim=True)

        if component == "u":
            return (Uc**2).mean()
        if component == "uv":
            return 0.5 * ((Uc**2).mean() + (Vc**2).mean())
        return (Vc**2).mean()


