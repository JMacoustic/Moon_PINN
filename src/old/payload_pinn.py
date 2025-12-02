import math, torch
import torch.nn as nn
import torch.autograd as autograd

# -------------- Elasticity utilities --------------
class Elastic2D:
    def __init__(self, mat):
        E, nu = float(mat.E), float(mat.nu)
        if getattr(mat, "plane_stress", True):
            self.lmbda = E * nu / (1.0 - nu**2)
            self.mu    = E / (2.0 * (1.0 + nu))
        else:
            self.lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.mu    = E / (2.0 * (1.0 + nu))
        self.rho = float(mat.rho)

    @torch.no_grad()
    def __repr__(self):
        return f"Elastic2D(lambda={self.lmbda:.3g}, mu={self.mu:.3g}, rho={self.rho:.3g})"

    def stress(self, dux, duy, dvx, dvy):
        # Return σxx, σyy, σxy (small strain, isotropic)
        exx = dux
        eyy = dvy
        exy = 0.5 * (duy + dvx)
        lam, mu = self.lmbda, self.mu
        tr = exx + eyy
        sxx = lam * tr + 2.0 * mu * exx
        syy = lam * tr + 2.0 * mu * eyy
        sxy = 2.0 * mu * exy
        return sxx, syy, sxy


# -------------- Small MLPs --------------
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=128, depth=6, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(in_dim, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class TimeMLP(nn.Module):
    """Scalar function y_p(t) for payload DOF."""
    def __init__(self, width=64, depth=3, act=nn.SiLU):
        super().__init__()
        layers = [nn.Linear(1, width), act()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), act()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        t = t.view(-1, 1)
        return self.net(t).view(-1)  # (N,)


# -------------- PINN with payload --------------
class ElastodynamicsPINN(nn.Module):
    """
    Expects `mesh` to supply:
      - verts_torch (nv,2), tris_torch (nt,3)
      - sample_interior(n), sample_on_nodes(ids,n) [optional for your sampling pipeline]
      - bottom_ids, top_ids  (1D Long tensors)
    Expects `cfg` to have:
      - T (float), f_base (Hz), a_base or y_base amplitude via cfg.V0 (displ amp)
      - payload: m_p, c_p, k_p (floats)
      - L_top (float): length of the top edge (for ∫σ_yy dΓ ≈ mean(σ_yy) * L_top)
      - optional: use_accel_metric: bool
    """
    def __init__(self, mesh, mat, cfg):
        super().__init__()
        self.mesh = mesh
        self.mat  = Elastic2D(mat)
        self.cfg  = cfg
        self.model = MLP(in_dim=3, out_dim=2, width=128, depth=6, act=nn.SiLU)
        self.payload = TimeMLP(width=64, depth=3, act=nn.SiLU)  # y_p(t)

    # call if your geometry changes
    def set_mesh(self, mesh):
        self.mesh = mesh

    # ----- helpers -----
    def forward(self, x, y, t):
        xyt = torch.stack([x, y, t], dim=-1)
        uv = self.model(xyt)
        return uv[..., 0], uv[..., 1]  # u(x,y,t), v(x,y,t)

    def _first_derivs(self, u, v, x, y, t):
        ones_u = torch.ones_like(u)
        ones_v = torch.ones_like(v)
        dux, duy, dut = autograd.grad(u, (x, y, t), grad_outputs=ones_u, create_graph=True, retain_graph=True)
        dvx, dvy, dvt = autograd.grad(v, (x, y, t), grad_outputs=ones_v, create_graph=True, retain_graph=True)
        return dux, duy, dut, dvx, dvy, dvt

    def _second_time(self, dtf, t):
        return autograd.grad(dtf, t, grad_outputs=torch.ones_like(dtf), create_graph=True, retain_graph=True)[0]

    # ----- PDE residual in the interior -----
    def pde_residual(self, x, y, t):
        x = x.requires_grad_(True); y = y.requires_grad_(True); t = t.requires_grad_(True)
        u, v = self.forward(x, y, t)
        dux, duy, dut, dvx, dvy, dvt = self._first_derivs(u, v, x, y, t)
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)
        dsxx_dx = autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True, retain_graph=True)[0]
        dsxy_dy = autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]
        dsyy_dy = autograd.grad(syy, y, grad_outputs=torch.ones_like(syy), create_graph=True, retain_graph=True)[0]
        dsxy_dx = autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]
        du_tt = self._second_time(dut, t)
        dv_tt = self._second_time(dvt, t)
        rx = self.mat.rho * du_tt - (dsxx_dx + dsxy_dy)
        ry = self.mat.rho * dv_tt - (dsxy_dx + dsyy_dy)
        return rx, ry

    # ----- Base excitation at bottom: v(x, y_btm, t) = y_base(t), u=0 -----
    def _y_base(self, t):
        # sinus base displacement (change to acceleration if you prefer)
        return self.cfg.V0 * torch.sin(2.0 * math.pi * self.cfg.f_base * t)

    def bc_bottom_base(self, x, y, t):
        u, v = self.forward(x, y, t)
        v_tgt = self._y_base(t)
        return u, v - v_tgt

    # ----- Payload coupling at top edge -----
    # 1) Kinematic compatibility: mean v on top == y_p(t)
    def payload_compatibility(self, t, x_top, y_top):
        yp = self.payload(t.requires_grad_(True))          # (N,)
        _, v = self.forward(x_top, y_top, t)               # (N,)
        v_mean = v.view(-1, 1).mean(dim=0)                 # (1,)
        # Make yp broadcastable with v_mean
        return (v_mean.squeeze() - yp).unsqueeze(0)        # residual ~ 0

    # 2) Dynamic balance on payload: m y'' + c y' + k(y - y_base) = F_struct->payload
    #    Approximate F_struct->payload ≈ - ∫_top σ_yy dΓ ≈ - mean(σ_yy) * L_top
    def payload_dynamics(self, t, x_top, y_top):
        t = t.requires_grad_(True)
        yp   = self.payload(t)                                         # (N,)
        yp_t = autograd.grad(yp, t, grad_outputs=torch.ones_like(yp), create_graph=True, retain_graph=True)[0]
        yp_tt= autograd.grad(yp_t, t, grad_outputs=torch.ones_like(yp_t), create_graph=True, retain_graph=True)[0]

        # evaluate σ_yy on the top edge
        x_top = x_top.requires_grad_(True); y_top = y_top.requires_grad_(True)
        u, v = self.forward(x_top, y_top, t)
        dux, duy, _, dvx, dvy, _ = self._first_derivs(u, v, x_top, y_top, t)
        _, syy, _ = self.mat.stress(dux, duy, dvx, dvy)
        syy_mean = syy.mean()

        # right-hand side: structural force acting on payload (upwards positive)
        F_struct = - syy_mean * float(self.cfg.L_top)

        # base displacement at the same t (scalar via mean)
        yb = self._y_base(t).mean()

        m, c, k = float(self.cfg.m_p), float(self.cfg.c_p), float(self.cfg.k_p)
        dyn_res = m * yp_tt + c * yp_t + k * (yp - yb) - F_struct
        return dyn_res

    # ----- Initial conditions (t=0): zero fields & payload at base -----
    def ic_zero_with_payload(self, x, y, t0):
        t0 = t0.requires_grad_(True)
        u, v = self.forward(x, y, t0)
        du_t = autograd.grad(u, t0, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv_t = autograd.grad(v, t0, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        yp   = self.payload(t0)
        yp_t = autograd.grad(yp, t0, grad_outputs=torch.ones_like(yp), create_graph=True)[0]
        yb0  = self._y_base(t0).mean()
        return u, v, du_t, dv_t, (yp - yb0), yp_t

    # ----- Transmissibility metric (RMS over time) -----
    def transmissibility_loss(self, time_steps: int = 33, use_accel: bool = False):
        with torch.enable_grad():
            device = next(self.parameters()).device
            times = torch.linspace(0.0, float(self.cfg.T), steps=time_steps, device=device)
            times.requires_grad_(True)

            yp   = self.payload(times)
            yp_t = autograd.grad(yp, times, grad_outputs=torch.ones_like(yp), create_graph=True, retain_graph=True)[0]
            yp_tt= autograd.grad(yp_t, times, grad_outputs=torch.ones_like(yp_t), create_graph=True, retain_graph=True)[0]

            yb   = self._y_base(times)
            yb_t = autograd.grad(yb, times, grad_outputs=torch.ones_like(yb), create_graph=True, retain_graph=True)[0]
            yb_tt= autograd.grad(yb_t, times, grad_outputs=torch.ones_like(yb_t), create_graph=True, retain_graph=True)[0]

            if use_accel or getattr(self.cfg, "use_accel_metric", False):
                num = yp_tt - yp_tt.mean()
                den = yb_tt - yb_tt.mean()
            else:
                num = yp   - yp.mean()
                den = yb   - yb.mean()

            eps = 1e-8
            ratio = num / (den.abs() + eps)
            return (ratio**2).mean()

    # ----- Convenience: get top/bottom node coords (as tensors on device) -----
    def _top_nodes_xy(self):
        device = next(self.parameters()).device
        B = self.mesh.verts_torch[self.mesh.top_ids].to(device)
        return B[:, 0], B[:, 1]

    def _bottom_nodes_xy(self):
        device = next(self.parameters()).device
        B = self.mesh.verts_torch[self.mesh.bottom_ids].to(device)
        return B[:, 0], B[:, 1]
