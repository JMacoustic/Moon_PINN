import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
import scipy.io
import scipy
import numpy as np
import torch
import torch.nn as nn
import os
import torch.nn.functional as F


def derivee(y,x):
  return autograd.grad(y,x,torch.ones_like(y),retain_graph=True,create_graph=True)[0]

def som(s):
  return torch.sum(s,dim=-1,keepdim=True)


class Offline:
    def __init__(self, device, mesh_data=None, time_steps=None):
        if mesh_data is None:
            raise ValueError("Mesh data must be provided for unsupervised PINN.")

        verts = torch.tensor(mesh_data['verts'], dtype=torch.float32).view(-1, 2).to(device)
        self.X = verts[:, 0:1]
        self.Y = verts[:, 1:2]

        # If time_steps is a scalar (e.g., 100), generate uniform time grid [0, 1]
        if isinstance(time_steps, int):
            self.T = torch.linspace(0, 1, time_steps).view(-1, 1).to(device)
        elif torch.is_tensor(time_steps):
            self.T = time_steps.to(device).view(-1, 1)
        else:
            raise ValueError("time_steps must be int or tensor")

        # Repeat mesh for each time step
        self.X = self.X.repeat(self.T.shape[0], 1)
        self.Y = self.Y.repeat(self.T.shape[0], 1)
        self.T = self.T.repeat_interleave(len(verts), dim=0)

        N = self.X.shape[0]
        self.L_pde = torch.arange(N, dtype=torch.float32, device=device)
        self.L_dbc = torch.zeros(0, dtype=torch.float32, device=device)
        self.L_nbc = torch.zeros(0, dtype=torch.float32, device=device)

        self.use_data = False

    def forward(self, idx, n, phi=False, dphi_dx=False, dphi_dy=False, d2phi_dx2=False, d2phi_dy2=False):
        x = self.X[idx.long()]
        y = self.Y[idx.long()]
        t = self.T[idx.long()]
        dummy = torch.zeros_like(x).repeat(1, n)
        outs = [x, y, t]
        if phi: outs.append(dummy)
        if dphi_dx: outs.append(100)
        if dphi_dy: outs.append(dummy)
        if d2phi_dx2: outs.append(dummy)
        if d2phi_dy2: outs.append(dummy)
        return tuple(outs)

class Online:
    def __init__(self, off, model, device):
        self.off = off
        self.model = model
        self.use_data = False  # Force unsupervised

        E, mu = 1.0, 0.3
        Cons = E/(1-mu**2)*torch.tensor([[1,mu,0],[mu,1,0],[0,0,0.5*(1-mu)]]).to(device)
        self.c1,self.c2,self.c3 = Cons[0,0],Cons[0,1],Cons[2,2]

    def label_pde(self):
        return self.off.L_pde

    def Nbc_loss(self, n_sigma):
        x, y, dphi_dx, dphi_dy = self.off.forward(self.off.L_nbc, n_sigma, dphi_dx=True, dphi_dy=True)
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, px, py = self.model.Fsigma(x,y, dphi_dx,dphi_dy)
        nbc_loss = F.mse_loss(sigma_xy, torch.zeros_like(sigma_xy)) + F.mse_loss(sigma_yy, x)
        return nbc_loss

    def Dbc_loss(self, n_u):
        x, y, phi = self.off.forward(self.off.L_dbc, n_u, phi=True)
        ux, uy = self.model(x, y, phi)
        dbc_loss = F.mse_loss(ux, torch.zeros_like(ux)) + F.mse_loss(uy, torch.zeros_like(uy))
        return dbc_loss

    def PDE_loss(self, label, n_sigma):
        x, y, t, dphi_dx, dphi_dy, d2phi_dx2, d2phi_dy2 = self.off.forward(label, n_sigma, dphi_dx=True, dphi_dy=True, d2phi_dx2=True, d2phi_dy2=True)
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, px, py = self.model.Fsigma(x, y, t, dphi_dx, dphi_dy)

        dsigmaxx_dx = derivee(sigma_xx, x) + som(px * d2phi_dx2)
        dsigmaxy_dy = derivee(sigma_xy, y) + som(px * d2phi_dy2)
        dsigmayx_dx = derivee(sigma_yx, x) + som(py * d2phi_dx2)
        dsigmayy_dy = derivee(sigma_yy, y) + som(py * d2phi_dy2)

        pde_x = dsigmaxx_dx + dsigmaxy_dy
        pde_y = dsigmayx_dx + dsigmayy_dy

        loss_x = F.mse_loss(pde_x, torch.zeros_like(pde_x))
        loss_y = F.mse_loss(pde_y, torch.zeros_like(pde_y))
        return loss_x + loss_y

    def C_loss(self,label, n_u, n_sigma):
        x, y, phi, dphi_dx, dphi_dy = self.off.forward(label, n_u, phi=True, dphi_dx=True, dphi_dy=True)
        x.requires_grad = True
        y.requires_grad = True
        phi.requires_grad = True

        ux, uy = self.model(x,y,phi)

        epsilon_xx = derivee(ux,x) + som(derivee(ux,phi) * dphi_dx)
        epsilon_xy = derivee(ux,y) + som(derivee(ux,phi) * dphi_dy)
        epsilon_yx = derivee(uy,x) + som(derivee(uy,phi) * dphi_dx)
        epsilon_yy = derivee(uy,y) + som(derivee(uy,phi) * dphi_dy)

        sigma_xx0 = self.c1*epsilon_xx + self.c2*epsilon_yy
        sigma_xy0 = self.c3*epsilon_xy + self.c3*epsilon_yx
        sigma_yx0 = self.c3*epsilon_xy + self.c3*epsilon_yx
        sigma_yy0 = self.c2*epsilon_xx + self.c1*epsilon_yy

        x, y, phi, dphi_dx, dphi_dy = self.off.forward(label, n_sigma, phi=True, dphi_dx=True, dphi_dy=True)
        sigma_xx, sigma_xy, sigma_yx, sigma_yy, H1, H2 = self.model.Fsigma(x,y,dphi_dx,dphi_dy)

        loss_C = F.mse_loss(torch.cat((sigma_xx0,sigma_xy0,sigma_yx0,sigma_yy0),dim=-1), \
                    torch.cat((sigma_xx,sigma_xy,sigma_yx,sigma_yy),dim=-1))

        return loss_C
    
class MLP_gelu(nn.Module):
    def __init__(self, insize, outsize, width, layers):
        super().__init__()
        self.input_layer = nn.Linear(insize, width)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(width, width) for _ in range(layers)
        ])
        self.output_layer = nn.Linear(width, outsize)

    def forward(self, x):
        x = F.gelu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.gelu(layer(x))
        return self.output_layer(x)

class Finite_PINN(nn.Module):
    def __init__(self, n_sigma, n_u):
        super(Finite_PINN, self).__init__()
        width = 128
        # Input now includes x, y, t (3 inputs)
        self.NN_H1 = MLP_gelu(3, n_sigma, width, 4)
        self.NN_H2 = MLP_gelu(3, n_sigma, width, 4)
        # Input now includes x, y, t + u vector (3 + n_u inputs)
        self.NN_eu1 = MLP_gelu(3 + n_u, 1, width, 4)
        self.NN_eu2 = MLP_gelu(3 + n_u, 1, width, 4)

    def forward(self, x, y, t, u):
        xtu = torch.cat((x, y, t, u), dim=-1)
        v1 = self.NN_eu1(xtu)
        v2 = self.NN_eu2(xtu)
        return v1, v2

    def Fsigma(self, x, y, t, du_dx, du_dy):
        xyt = torch.cat((x, y, t), dim=-1)
        px = self.NN_H1(xyt)
        py = self.NN_H2(xyt)
        sig_xx = som(px * du_dx)
        sig_xy = som(px * du_dy)
        sig_yx = som(py * du_dx)
        sig_yy = som(py * du_dy)
        return sig_xx, sig_xy, sig_yx, sig_yy, px, py