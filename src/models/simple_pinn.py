"""
Auxetic PINN for linear elastodynamics in 2D (plane stress/strain).
Assumes you already have `generate_auxetic(...)` that returns a dict with
- verts: flat list [x0,y0, x1,y1, ...]
- triIds: flat list [i0,i1,i2, i3,i4,i5, ...]
- edgeIds: [(a,b), ...]
- bcIds: indices of vertices on the top boundary
- lattice_points: (ny,nx,2) array (optional)


This script builds a PINN that predicts displacement field (u, v) = f(x,y,t).
PDE enforced: rho * d²u/dt² = ∂sigma_xx/∂x + ∂sigma_xy/∂y + b_x
rho * d²v/dt² = ∂sigma_yx/∂x + ∂sigma_yy/∂y + b_y
with linear elasticity (isotropic), plane stress by default.


Boundary conditions used here (you can edit easily):
- Bottom edge (y=min) clamped: u=v=0.
- Side edges traction-free by default.
- Top edge (y=max) prescribed vertical harmonic displacement: u=0, v=V0*sin(2π f t).


Outputs:
- Trains the PINN.
- Saves snapshots of nodal displacements over time to `outputs/pinn_disp_t####.npz`.


Note:
- PINNs are sensitive to scaling; use geometry/time scaling if needed.
- This is a clean starting point; tune batch sizes, network depth, and loss weights.
"""


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from src.utils.auxetic import *
from typing import Dict

# -------------- Helpers: mesh, sampling --------------
class Mesh:
    def __init__(self, aux: Dict):
        verts = np.asarray(aux["verts"], dtype=float).reshape(-1,2)
        tris = np.asarray(aux["triIds"], dtype=int).reshape(-1,3)
        self.verts = torch.tensor(verts, dtype=torch.float32)
        self.tris = torch.tensor(tris, dtype=torch.long)
        self.nv = self.verts.shape[0]
        self.nt = self.tris.shape[0]
        
        # boundaries: detect bottom/top by y-extrema
        y = self.verts[:,1].numpy()
        self.ymin, self.ymax = float(y.min()), float(y.max())
        tol = 1e-9
        self.bottom_ids = torch.nonzero(torch.isclose(self.verts[:,1], torch.tensor(self.ymin), atol=tol), as_tuple=False).squeeze(1)
        self.top_ids = torch.nonzero(torch.isclose(self.verts[:,1], torch.tensor(self.ymax), atol=tol), as_tuple=False).squeeze(1)
        
        # precompute triangle areas & vertex coords per tri for sampling
        v = self.verts
        I = self.tris
        A = 0.5 * torch.abs(
            (v[I[:,1],0]-v[I[:,0],0])*(v[I[:,2],1]-v[I[:,0],1]) -
            (v[I[:,2],0]-v[I[:,0],0])*(v[I[:,1],1]-v[I[:,0],1])
        )
        self.areas = A
        self.total_area = float(A.sum())
        self.alias_probs = (A / A.sum()).numpy()

    def sample_interior(self, n: int) -> torch.Tensor:
        """Sample `n` points uniformly by area inside triangles via barycentric sampling."""
        idx = np.random.choice(self.nt, size=n, p=self.alias_probs)
        tri = self.tris[idx]  # (n,3)
        A = self.verts[tri[:,0]]
        B = self.verts[tri[:,1]]
        C = self.verts[tri[:,2]]
        r1 = torch.rand(n,1)
        r2 = torch.rand(n,1)
        s = torch.sqrt(r1)
        l1 = 1.0 - s
        l2 = s * (1.0 - r2)
        l3 = s * r2
        P = l1*A + l2*B + l3*C
        return P

    def sample_on_nodes(self, ids: torch.Tensor, n: int) -> torch.Tensor:
        """Sample nodes (x,y) from a list of indices uniformly with replacement."""
        if ids.numel() == 0:
            return torch.empty(0,2)
        choice = torch.randint(0, ids.numel(), (n,))
        sel = ids[choice]
        return self.verts[sel]

# -------------- Elasticity utilities --------------
class Elastic2D:
    def __init__(self, mat):
        E, nu = mat.E, mat.nu
        if mat.plane_stress:
            # plane stress
            self.lmbda = E*nu/(1-nu**2)
            self.mu = E/(2*(1+nu))
        else:
            # plane strain
            self.lmbda = E*nu/((1+nu)*(1-2*nu))
            self.mu = E/(2*(1+nu))
        self.rho = mat.rho

    def stress(self, dux, duy, dvx, dvy):
        """Return σxx, σyy, σxy for small-strain isotropic elasticity."""
        exx = dux
        eyy = dvy
        exy = 0.5*(duy + dvx)
        lam, mu = self.lmbda, self.mu
        tr = exx + eyy
        sxx = lam*tr + 2*mu*exx
        syy = lam*tr + 2*mu*eyy
        sxy = 2*mu*exy
        return sxx, syy, sxy

# -------------- PINN model --------------
class MLP(nn.Module):
    def __init__(self, in_dim=3, out_dim=2, width=128, depth=6, act=nn.Tanh):
        super().__init__()
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(act())
        for _ in range(depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(act())
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, xyt):
        return self.net(xyt)

# -------------- Loss terms --------------
class ElastodynamicsPINN(nn.Module):
    def __init__(self, mesh, mat, cfg):
        super().__init__()
        self.mesh = mesh
        self.mat = Elastic2D(mat)
        self.cfg = cfg
        self.model = MLP(in_dim=3, out_dim=2, width=128, depth=6, act=nn.SiLU)


    def forward(self, x, y, t):
        xyt = torch.stack([x,y,t], dim=-1)
        uv = self.model(xyt)
        return uv[...,0], uv[...,1]


    def pde_residual(self, x, y, t):
        # ensure grads for all coords
        x = x.requires_grad_(True)
        y = y.requires_grad_(True)
        t = t.requires_grad_(True)
        u, v = self.forward(x, y, t)


        # spatial grads (compute each explicitly)
        dux = autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        duy = autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dvx = autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        dvy = autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        sxx, syy, sxy = self.mat.stress(dux, duy, dvx, dvy)


        # divergence of stress
        dsxx_dx = autograd.grad(sxx, x, grad_outputs=torch.ones_like(sxx), create_graph=True, retain_graph=True)[0]
        dsxy_dy = autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]
        dsyy_dy = autograd.grad(syy, y, grad_outputs=torch.ones_like(syy), create_graph=True, retain_graph=True)[0]
        dsxy_dx = autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True, retain_graph=True)[0]


        # time accel
        du_t = autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv_t = autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        du_tt = autograd.grad(du_t, t, grad_outputs=torch.ones_like(du_t), create_graph=True)[0]
        dv_tt = autograd.grad(dv_t, t, grad_outputs=torch.ones_like(dv_t), create_graph=True)[0]


        # body force (zero here)
        bx = 0.0
        by = 0.0


        rx = self.mat.rho*du_tt - (dsxx_dx + dsxy_dy) - bx
        ry = self.mat.rho*dv_tt - (dsxy_dx + dsyy_dy) - by
        return rx, ry


    def bc_bottom_clamp(self, x, y, t):
        u, v = self.forward(x, y, t)
        return u, v


    def bc_top_disp(self, x, y, t):
        u, v = self.forward(x, y, t)
        u_tgt = torch.zeros_like(u)
        v_tgt = self.cfg.V0 * 4*torch.sin(2*math.pi*self.cfg.f_top*t)
        return u - u_tgt, v - v_tgt


    def ic_zero(self, x, y, t0):
        # make sure time carries gradient for velocity ICs
        t0 = t0.requires_grad_(True)
        u, v = self.forward(x, y, t0)
        du_t = autograd.grad(u, t0, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        dv_t = autograd.grad(v, t0, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        # u(x,y,0)=0, v(x,y,0)=0, and velocities 0
        return u, v, du_t, dv_t
    
    
    def bottom_vibration_loss(self, time_steps: int = 25, component: str = "v"):
        """
        Penalize vibration amplitude on the bottom edge.
        - time_steps: number of time samples in [0, T]
        - component: 'u' | 'v' | 'uv'
            'u'  -> horizontal only
            'v'  -> vertical only (default)
            'uv' -> mean of both components
        Returns a scalar loss (mean squared, time-RMS with DC removed).
        """
        # no bottom nodes -> zero loss
        if self.mesh.bottom_ids.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=next(self.parameters()).device)

        device = next(self.parameters()).device
        # bottom-node coordinates on the right device
        B = self.mesh.verts[self.mesh.bottom_ids].to(device)   # (Nb, 2)
        xB, yB = B[:, 0], B[:, 1]

        # sample times uniformly over [0, T]
        times = torch.linspace(0.0, self.cfg.T, steps=time_steps, device=device)

        u_list, v_list = [], []
        for tk in times:
            tB = torch.full_like(xB, fill_value=float(tk))
            uB, vB = self.forward(xB, yB, tB)   # (Nb,), (Nb,)
            u_list.append(uB)
            v_list.append(vB)

        U = torch.stack(u_list, dim=0)  # (T, Nb)
        V = torch.stack(v_list, dim=0)  # (T, Nb)

        # remove DC per node to measure true vibration amplitude
        Uc = U - U.mean(dim=0, keepdim=True)
        Vc = V - V.mean(dim=0, keepdim=True)

        if component == "u":
            loss = (Uc**2).mean()
        elif component == "uv":
            loss = 0.5 * ((Uc**2).mean() + (Vc**2).mean())
        else:  # 'v' (default)
            loss = (Vc**2).mean()

        return loss


    # (train_pinn and __main__ unchanged except they now call this updated PINN)

