import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_tri_strain(xi, yi, xj, yj, xm, ym, ui, vi, uj, vj, um, vm):
    A = xi * ( yj - ym ) + xj * ( ym - yi ) + xm * ( yi - yj )
    ai = xj*ym - yj*xm
    aj = xm*yi - ym*xi
    am = xi*yj - yi*xj
    bi = yi - ym
    bj = ym - yi
    bm = yi - yj
    gi = xm - xj
    gj = xi - xm
    gm = xj - xi

    exx = (bi*ui + bj*uj + bm*um) / 2 / A
    eyy = (gi*vi + gj*vj + gm*vm) / 2 / A
    exy = (gi*ui + gj*uj + gm*um + bi*vi + bj*vj + bm*vm) / 2 / A

    return exx, eyy, exy


def get_tri_stress(exx, eyy, exy, E, nu):
    k = E / (1 - nu**2)
    sxx = (exx + nu*eyy) / k
    syy = (nu * exx + eyy) / k
    sxy = exy * (1 - nu) / 2 / k

    return sxx, syy, sxy


class Offline:
    def __init__(self, device, mesh_data, E, nu, t, rho, endtime, steps):
        self.E = E
        self.nu = nu
        self.t = t
        self.rho = rho
        self.device = device
        self.vertices = torch.tensor(mesh_data['verts'], dtype=torch.float32).view(-1, 2).to(device)
        self.tri_Ids = torch.tensor(mesh_data['triIds'], dtype=torch.int).view(-1, 3).to(device)
        self.edge_Ids = torch.tensor(mesh_data['edgeIds'], dtype=torch.int).to(device)
        self.bc_Ids = torch.tensor(mesh_data['bcIds'], dtype=torch.int).to(device)

        self.T = torch.linspace(0, endtime, steps).to(device)
        self.dt = endtime/steps

        self.X = self.vertices[:, 0]
        self.Y = self.vertices[:, 1]

        self.X_bc = self.X[self.bc_Ids]
        self.X_bc = self.X_bc.repeat(1, self.T.shape[0])
        self.Y_bc = 0.0 * torch.sin(2*3.14*self.T).repeat_interleave(len(self.bc_Ids)).to(device)
        self.X = self.X.repeat(self.T.shape[0], 1)
        self.Y = self.Y.repeat(self.T.shape[0], 1)

        self.K, self.M = self.assemble_global_mat()

    def assemble_global_mat(self):
        N = self.vertices.shape[0]
        K_global = torch.zeros((2*N, 2*N), dtype=torch.float32, device=self.device)
        M_global = torch.zeros((2*N, 2*N), dtype=torch.float32, device=self.device)

        for tri in self.tri_Ids:
            i, j, m = tri.tolist()
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            xm, ym = self.vertices[m]

            K_local = self.get_stiffness_mat(self.t, self.E, self.nu, xi, yi, xj, yj, xm, ym)
            M_local = self.get_mass_mat(self.rho, xi, yi, xj, yj, xm, ym)

            dof_map = [
                2*i, 2*i+1,
                2*j, 2*j+1,
                2*m, 2*m+1]
            
            for a in range(6):
                for b in range(6):
                    K_global[dof_map[a], dof_map[b]] += K_local[a, b]
                    M_global[dof_map[a], dof_map[b]] += M_local[a, b]
        
        return K_global, M_global
    
    def get_stiffness_mat(self, t, E, nu, xi, yi, xj, yj, xm, ym):
        A = (xi * ( yj - ym ) + xj * ( ym - yi ) + xm * ( yi - yj )) / 2

        bi = yi - ym
        bj = ym - yi
        bm = yi - yj
        gi = xm - xj
        gj = xi - xm
        gm = xj - xi

        B = torch.tensor([[bi, 0, bj, 0, bm, 0],
                        [0, gi, 0, gj, 0, gm],
                        [gi, bi, gj, bj, gm, bm]], dtype=torch.float32, device=self.device) / 2 / A
        
        D = torch.tensor([[1, nu, 0],
                        [nu, 1, 0],
                        [0, 0, (1-nu)/2]], dtype=torch.float32, device=self.device) * E / (1-nu**2)
        
        

        K = t * A * B.T @ D @ B

        return K

    def get_mass_mat(self, rho, xi, yi, xj, yj, xm, ym):
        A = (xi * ( yj - ym ) + xj * ( ym - yi ) + xm * ( yi - yj )) / 2
        M = torch.eye(6).to(self.device) * rho * A / 3

        return M



class Online:
    def __init__(self, off, model, device):
        self.off = off
        self.model = model
        self.device = device

    def PDE_loss(self, u_pred):
        """
        u_pred: shape (T, 2N), predicted displacements over time
        """
        dt = self.off.dt
        M = self.off.M.to(self.device)
        K = self.off.K.to(self.device)
        g = 9.8

        u_prev = u_pred[:-2]
        u_curr = u_pred[1:-1]
        u_next = u_pred[2:]

        # Estimate acceleration
        acc = (u_next - 2 * u_curr + u_prev) / (dt ** 2)  # (T-2, 2N)

        # Gravity vector [0, -g, 0, -g, ..., 0, -g] of shape (2N,)
        N = u_pred.shape[1] // 2
        gravity_vec = torch.zeros((2 * N,), device=self.device)
        gravity_vec[1::2] = -g
        f_ext = M @ gravity_vec 

        residuals = torch.einsum('ij,tj->ti', M, acc) + torch.einsum('ij,tj->ti', K, u_curr) - f_ext
        loss = torch.mean(residuals ** 2)

        return loss
    
    def BC_loss(self, u_pred):
        T = self.off.T.shape[0]
        N = self.off.vertices.shape[0]
        bc_ids = self.off.bc_Ids

        # Reshape to [T, N, 2]
        u_reshaped = u_pred.view(T, N, 2)

        u_x_pred = u_reshaped[:, bc_ids, 0].reshape(-1)
        u_y_pred = u_reshaped[:, bc_ids, 1].reshape(-1)

        u_x_target = self.off.X_bc
        u_y_target = self.off.Y_bc

        loss_x = torch.mean((u_x_pred - u_x_target) ** 2)
        loss_y = torch.mean((u_y_pred - u_y_target) ** 2)
        return loss_y
    
    def IC_loss(self, u_pred):
        u_0 = u_pred[0]       # [2N]
        u_1 = u_pred[1]       # [2N]
        dt = self.off.dt

        loss_disp = torch.mean(u_0 ** 2)
        loss_vel = torch.mean(((u_1 - u_0) / dt) ** 2)

        return loss_disp + loss_vel


        
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
    def __init__(self):
        super(Finite_PINN, self).__init__()
        width = 128
        self.network = MLP_gelu(3, 2, width, 4)  # input: x, y, t → output: u_x, u_y

    def forward(self, x, y, t):
        """
        Input: x, y, t (each of shape [B, 1] or [B])
        Output: u_x, u_y (each [B, 1])
        """
        xyt = torch.cat((x, y, t), dim=-1)
        u = self.network(xyt)
        return u[:, 0:1], u[:, 1:2]  # u_x, u_y
    

class NN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.acts = nn.ModuleList()

        for i in range(1, num_layers + 1):
            if i < num_layers:
                out_dim_ = hidden_dim
            else:
                out_dim_ = out_dim

            fc = nn.Linear(in_dim, out_dim_)

            self.fcs.append(fc)
            in_dim = hidden_dim

            if i < num_layers:
                self.acts.append(nn.Tanh())

    def forward(self, x):
        for fc, act in zip(self.fcs, self.acts):
          x = act(fc(x))
        return self.fcs[-1](x)