from torch import autograd
import torch

def uv_to_vec(uv):
    # uv: (Nn, 2) -> q: (2*Nn,)
    return uv.reshape(-1)

def vec_to_uv(q):
    # q: (2*Nn,) -> uv: (Nn, 2)
    Nn2 = q.numel()
    assert Nn2 % 2 == 0
    return q.view(-1, 2)

def verts_to_vec(uv):
    # uv: (..., N_verts, 2) -> (..., 2*N_verts)
    last_two = uv.shape[-2:]
    assert last_two[1] == 2
    return uv.reshape(*uv.shape[:-2], -1)

def verts_force_to_vec(f_verts):
    # (N_verts, 2) -> (2*N_verts,)
    return f_verts.reshape(-1)

def build_cst_mk(verts, tris, E, nu, rho, th, device=None, dtype=torch.float32):
    """
    verts: (Nn, 2) float tensor [x, y]
    tris:  (Ne, 3) long tensor [i0, i1, i2]
    returns: M, K of shape (2*Nn, 2*Nn)
    Vectorized (no Python loop over elements).
    """
    if device is None:
        device = verts.device

    verts = verts.to(device=device, dtype=dtype)
    tris  = tris.to(device=device, dtype=torch.long)

    Nn   = verts.shape[0]
    Ndof = 2 * Nn
    Ne   = tris.shape[0]

    # --- plane stress D ---
    coeff = E / (1.0 - nu**2)
    D = coeff * torch.tensor(
        [[1.0,    nu,          0.0],
         [nu,     1.0,         0.0],
         [0.0,    0.0, (1.0 - nu) / 2.0]],
        device=device, dtype=dtype
    )  # (3,3)

    # --- nodal indices per element ---
    i0 = tris[:, 0]
    i1 = tris[:, 1]
    i2 = tris[:, 2]

    # node coordinates per element: (Ne, 2)
    p1 = verts[i0]
    p2 = verts[i1]
    p3 = verts[i2]

    x1, y1 = p1[:, 0], p1[:, 1]
    x2, y2 = p2[:, 0], p2[:, 1]
    x3, y3 = p3[:, 0], p3[:, 1]

    # area A (Ne,)
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * detJ  # keep sign convention as in your original code

    # b, c coefficients (Ne,)
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1

    inv_2A = 1.0 / (2.0 * A)

    # --- B matrices for all elements: (Ne, 3, 6) ---
    z = torch.zeros_like(b1)

    row1 = torch.stack([b1, z,  b2, z,  b3, z], dim=1)  # (Ne, 6)
    row2 = torch.stack([z,  c1, z,  c2, z,  c3], dim=1)
    row3 = torch.stack([c1, b1, c2, b2, c3, b3], dim=1)

    B = inv_2A.view(-1, 1, 1) * torch.stack([row1, row2, row3], dim=1)  # (Ne, 3, 6)

    # --- element stiffness Ke: (Ne, 6, 6) ---
    # Ke_e = th * A_e * B^T D B
    BD  = torch.matmul(D.expand(Ne, -1, -1), B)           # (Ne, 3, 6)
    Ke  = th * A.view(Ne, 1, 1) * torch.matmul(B.transpose(1, 2), BD)

    # --- element mass Me: (Ne, 6, 6) ---
    M_template = torch.tensor(
        [[2, 0, 1, 0, 1, 0],
         [0, 2, 0, 1, 0, 1],
         [1, 0, 2, 0, 1, 0],
         [0, 1, 0, 2, 0, 1],
         [1, 0, 1, 0, 2, 0],
         [0, 1, 0, 1, 0, 2]],
        device=device, dtype=dtype
    )  # (6,6)

    Me = (rho * th * A.view(Ne, 1, 1) / 12.0) * M_template  # (Ne,6,6), A broadcast

    # --- global DOF indices per element: (Ne, 6) ---
    dofs = torch.stack([
        2 * i0, 2 * i0 + 1,
        2 * i1, 2 * i1 + 1,
        2 * i2, 2 * i2 + 1
    ], dim=1)  # (Ne, 6)

    # --- allocate global matrices ---
    M = torch.zeros((Ndof, Ndof), device=device, dtype=dtype)
    K = torch.zeros((Ndof, Ndof), device=device, dtype=dtype)

    # --- assemble with index_put_ ---
    # rows, cols: (Ne*6*6,)
    rows = dofs.unsqueeze(2).expand(-1, 6, 6).reshape(-1)
    cols = dofs.unsqueeze(1).expand(-1, 6, 6).reshape(-1)

    M_vals = Me.reshape(-1)
    K_vals = Ke.reshape(-1)

    M.index_put_((rows, cols), M_vals, accumulate=True)
    K.index_put_((rows, cols), K_vals, accumulate=True)

    return M, K

def time_derivatives_batch(model, t_batch, N_verts):
    """
    model: t (scalar) -> (N_verts, 2)
    t_batch: 1D tensor (N_t,)
    returns:
        q      : (N_t, 2*N_verts)
        qdot   : (N_t, 2*N_verts)
        qddot  : (N_t, 2*N_verts)
    """
    device = t_batch.device
    q_list, qdot_list, qddot_list = [], [], []

    for tj in t_batch:
        t = tj.clone().detach().to(device).requires_grad_(True)

        def f_scalar(tau):
            uv = model(tau)        # (N_verts, 2)
            return flatten_uv(uv)  # (2*N_verts,)

        # q(t)
        q = f_scalar(t)                    # (2*N_verts,)

        # dq/dt via Jacobian wrt scalar t  -> (2*N_verts,)
        qdot = jacobian(f_scalar, t, create_graph=True)  # shape (2*N_verts,)

        # d²q/dt²: Jacobian of qdot wrt t  -> (2*N_verts,)
        def g_scalar(tau):
            return jacobian(f_scalar, tau, create_graph=True)

        qddot = jacobian(g_scalar, t, create_graph=True)

        q_list.append(q)
        qdot_list.append(qdot)
        qddot_list.append(qddot)

    q     = torch.stack(q_list, dim=0)      # (N_t, 2*N_verts)
    qdot  = torch.stack(qdot_list, dim=0)   # (N_t, 2*N_verts)
    qddot = torch.stack(qddot_list, dim=0)  # (N_t, 2*N_verts)
    return q, qdot, qddot


def energy_residual_batch_fourier(t_batch, model, M, Kmat, f_verts, C=None):
    """
    t_batch: (N_t,)
    model.eval_with_derivs(t) -> (disp, vel, acc) = (..., N_verts, 2)
    """
    device = M.device
    t_batch = t_batch.to(device)

    disp, vel, acc = model.eval_with_derivs(t_batch)  # (N_t, N_verts, 2)
    q     = verts_to_vec(disp)                        # (N_t, 2N)
    qdot  = verts_to_vec(vel)
    qddot = verts_to_vec(acc)

    f_ext_vec = verts_force_to_vec(f_verts).to(device)

    term_kin  = torch.einsum('bi,ij,bj->b', qdot, M,    qddot)
    term_str  = torch.einsum('bi,ij,bj->b', qdot, Kmat, q)
    if C is not None:
        term_diss = torch.einsum('bi,ij,bj->b', qdot, C, qdot)
    else:
        term_diss = torch.zeros_like(term_kin)

    term_ext  = torch.einsum('i,bi->b', f_ext_vec, qdot)
    rE = term_kin + term_str + term_diss - term_ext
    return rE


def energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None):
    rE = energy_residual_batch_fourier(t_batch, model, M, Kmat, f_verts, C=C)
    return torch.mean(rE**2)


def loss_initial_condition(model, u0_verts: torch.Tensor):
    """
    Initial condition loss:
      disp(verts, t=0) ≈ u0_verts

    model     : FourierNodeModel
    u0_verts  : (N_verts, 2) tensor of target initial displacements
    returns   : scalar loss
    """
    device = u0_verts.device
    t0 = torch.tensor(0.0, device=device)

    disp0, _, _ = model.eval_with_derivs(t0)   # (N_verts, 2)
    loss_ic = torch.mean((disp0 - u0_verts)**2)
    return loss_ic


def loss_boundary_sine(
    model,
    t_batch: torch.Tensor,
    bc_ids: torch.Tensor,
    amp_y: float,
    phase: float = 0.0,
    offset_y: float = 0.0,
    x_fixed: float | None = 0.0,
):
    """
    Boundary condition loss:
      for verts in bc_ids, enforce
        v(t) ≈ offset_y + amp_y * sin(omega * t + phase)
      optionally enforce u(t) ≈ x_fixed.

    model   : FourierNodeModel (must have .omega and .eval_with_derivs)
    t_batch : (N_t,) time samples
    bc_ids  : (N_bc,) long tensor of boundary vert indices
    amp_y   : sine amplitude
    phase   : phase shift [rad]
    offset_y: vertical offset
    x_fixed : if not None, target x-displacement for those verts
    """
    device = t_batch.device
    t_batch = t_batch.to(device)
    bc_ids = bc_ids.to(device)

    # (N_t, N_verts, 2)
    disp, _, _ = model.eval_with_derivs(t_batch)

    # predicted boundary displacements
    u_bc = disp[:, bc_ids, 0]   # (N_t, N_bc)
    v_bc = disp[:, bc_ids, 1]   # (N_t, N_bc)

    # target y(t) for boundary verts
    omega = model.omega
    wt = omega * t_batch.view(-1, 1)  # (N_t, 1)
    v_target = offset_y + amp_y * torch.sin(wt + phase)  # (N_t, 1)
    v_target = v_target.expand_as(v_bc)                  # (N_t, N_bc)

    loss_v = torch.mean((v_bc - v_target)**2)

    if x_fixed is not None:
        u_target = torch.full_like(u_bc, float(x_fixed))
        loss_u = torch.mean((u_bc - u_target)**2)
        loss_bc = loss_u + loss_v
    else:
        loss_bc = loss_v

    return loss_bc


def loss_bottom_vibration(
    model,
    mesh,
    T: float,
    time_steps: int = 25,
    component: str = "v",
):
    """
    Time-RMS (DC-removed) of bottom-edge motion.
    No nondimensionalization.

    Args
    ----
    model      : FourierNodeModel (must have eval_with_derivs(t))
    mesh       : Aux (must have verts_torch and bottom_ids)
    T          : final time for evaluation (same as train_cfg.T)
    time_steps : number of time samples over [0, T]
    component  : "u", "v", or "uv"

    Returns
    -------
    loss : scalar tensor
        Mean squared RMS of chosen component(s) over bottom verts and time.
    """
    device = next(model.parameters()).device

    # no bottom nodes -> zero loss
    if mesh.bottom_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)
    Nb = bottom_ids.numel()

    # time grid (N_t,)
    times = torch.linspace(0.0, T, steps=time_steps, device=device)

    # model eval: disp: (N_t, N_verts, 2)
    disp, _, _ = model.eval_with_derivs(times)

    # bottom displacements: (N_t, Nb)
    U = disp[:, bottom_ids, 0]   # x-displacement
    V = disp[:, bottom_ids, 1]   # y-displacement

    # remove DC per node over time
    Uc = U - U.mean(dim=0, keepdim=True)   # (N_t, Nb)
    Vc = V - V.mean(dim=0, keepdim=True)

    if component == "u":
        return (Uc**2).mean()
    elif component == "v":
        return (Vc**2).mean()
    elif component == "uv":
        return 0.5 * ((Uc**2).mean() + (Vc**2).mean())
    else:
        # default to vertical if unknown component
        return (Vc**2).mean()