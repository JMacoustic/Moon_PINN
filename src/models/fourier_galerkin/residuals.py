import torch


def uv_to_vec(uv: torch.Tensor) -> torch.Tensor:
    # uv: (Nn, 2) -> q: (2*Nn,)
    return uv.reshape(-1)


def vec_to_uv(q: torch.Tensor) -> torch.Tensor:
    # q: (2*Nn,) -> uv: (Nn, 2)
    Nn2 = q.numel()
    assert Nn2 % 2 == 0
    return q.view(-1, 2)


def verts_to_vec(uv: torch.Tensor) -> torch.Tensor:
    # uv: (..., N_verts, 2) -> (..., 2*N_verts)
    last_two = uv.shape[-2:]
    assert last_two[1] == 2
    return uv.reshape(*uv.shape[:-2], -1)


def verts_force_to_vec(f_verts: torch.Tensor) -> torch.Tensor:
    # (N_verts, 2) -> (2*N_verts,)
    return f_verts.reshape(-1)


def build_cst_mk(verts, tris, E, nu, rho, z_width, device=None, dtype=torch.float32):
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
        [
            [1.0,    nu,          0.0],
            [nu,     1.0,         0.0],
            [0.0,    0.0, (1.0 - nu) / 2.0],
        ],
        device=device,
        dtype=dtype,
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
    A = 0.5 * detJ  # sign convention same as original

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
    BD  = torch.matmul(D.expand(Ne, -1, -1), B)           # (Ne, 3, 6)
    Ke  = z_width * A.view(Ne, 1, 1) * torch.matmul(B.transpose(1, 2), BD)

    # --- element mass Me: (Ne, 6, 6) ---
    M_template = torch.tensor(
        [
            [2, 0, 1, 0, 1, 0],
            [0, 2, 0, 1, 0, 1],
            [1, 0, 2, 0, 1, 0],
            [0, 1, 0, 2, 0, 1],
            [1, 0, 1, 0, 2, 0],
            [0, 1, 0, 1, 0, 2],
        ],
        device=device,
        dtype=dtype,
    )  # (6,6)

    Me = (rho * z_width * A.view(Ne, 1, 1) / 12.0) * M_template  # (Ne,6,6)

    # --- global DOF indices per element: (Ne, 6) ---
    dofs = torch.stack(
        [
            2 * i0, 2 * i0 + 1,
            2 * i1, 2 * i1 + 1,
            2 * i2, 2 * i2 + 1,
        ],
        dim=1,
    )  # (Ne, 6)

    # --- allocate global matrices ---
    M = torch.zeros((Ndof, Ndof), device=device, dtype=dtype)
    K = torch.zeros((Ndof, Ndof), device=device, dtype=dtype)

    # --- assemble with index_put_ ---
    rows = dofs.unsqueeze(2).expand(-1, 6, 6).reshape(-1)
    cols = dofs.unsqueeze(1).expand(-1, 6, 6).reshape(-1)

    M_vals = Me.reshape(-1)
    K_vals = Ke.reshape(-1)

    M.index_put_((rows, cols), M_vals, accumulate=True)
    K.index_put_((rows, cols), K_vals, accumulate=True)

    return M, K


def energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None):
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

    term_kin  = torch.einsum("bi,ij,bj->b", qdot, M,    qddot)
    term_str  = torch.einsum("bi,ij,bj->b", qdot, Kmat, q)
    if C is not None:
        term_diss = torch.einsum("bi,ij,bj->b", qdot, C, qdot)
    else:
        term_diss = torch.zeros_like(term_kin)

    term_ext  = torch.einsum("i,bi->b", f_ext_vec, qdot)
    rE = term_kin + term_str + term_diss - term_ext
    return torch.mean(rE**2)


# def energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None):
#     rE = energy_residual_batch_fourier(t_batch, model, M, Kmat, f_verts, C=C)
#     return torch.mean(rE**2)


def loss_initial_condition(model, u0_verts: torch.Tensor):
    """
    Initial condition loss:
      disp(verts, t=0) ≈ u0_verts
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
    """
    device = t_batch.device
    t_batch = t_batch.to(device)
    bc_ids = bc_ids.to(device)

    disp, _, _ = model.eval_with_derivs(t_batch)  # (N_t, N_verts, 2)

    u_bc = disp[:, bc_ids, 0]   # (N_t, N_bc)
    v_bc = disp[:, bc_ids, 1]   # (N_t, N_bc)

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


# def loss_bottom_vibration(
#     model,
#     mesh,
#     component: str = "v",
# ):
#     """
#     Time-RMS (DC-removed) of bottom-edge motion, computed analytically
#     from Fourier coefficients (no time sampling).

#     Assumes:
#       - model._coeffs() returns (coef_u, coef_v) with layout
#             [a0, a1, b1, a2, b2, ..., aK, bK],
#       - all cosine terms a0, a_k have been zeroed in _coeffs()
#         (so time-mean ≈ 0 and RMS^2 = 0.5 * sum_k b_k^2).
#     """
#     device = next(model.parameters()).device

#     if mesh.bottom_ids.numel() == 0:
#         return torch.zeros((), dtype=torch.float32, device=device)

#     bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)

#     coef_u, coef_v = model._coeffs()  # (N_verts, 2K+1)
#     coef_u_b = coef_u[bottom_ids, :]
#     coef_v_b = coef_v[bottom_ids, :]

#     b_u = coef_u_b[:, 2::2]  # (Nb, K)
#     b_v = coef_v_b[:, 2::2]

#     if component == "u":
#         return torch.sqrt(0.5 * (b_u**2).mean())
#     elif component == "v":
#         return torch.sqrt(0.5 * (b_v**2).mean())
#     elif component == "uv":
#         return torch.sqrt(0.25 * ((b_u**2).mean() + (b_v**2).mean()))
#     else:
#         return torch.sqrt(0.5 * (b_v**2).mean())


def loss_bottom_vibration(model, mesh, component: str = "v"):
    """
    Sum of per-mode amplitudes averaged over bottom nodes.

        R_i,k = sqrt(a_i,k^2 + b_i,k^2)
        mean_R_k = mean over bottom nodes
        loss = sum_k mean_R_k
    """

    device = next(model.parameters()).device
    if mesh.bottom_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)
    coef_u, coef_v = model._coeffs() 

    coef = coef_u if component == "u" else coef_v
    coef_b = coef[bottom_ids, :] 

    a = coef_b[:, 1::2]                # (Nb, K)
    b = coef_b[:, 2::2]                # (Nb, K)
    R = torch.sqrt(a * a + b * b)      # (Nb, K)

    mean_R_per_mode = R.mean(dim=0)
    return mean_R_per_mode.sum()


def loss_collision(mesh, margin: float = 0.001, eps: float = 1e-6):
    """
    Penalize violations of:
        px - 2*x_offset - t > margin
        py > margin, x_offset > margin, t > margin
    """
    # assume these are nn.Parameter / tensors
    px = mesh.pitch[0]
    py = mesh.pitch[1]
    xoff = mesh.x_offset
    t = mesh.thickness

    loss = torch.zeros((), dtype=px.dtype, device=px.device)
    
    gap = px - 2.0 * xoff - t
    loss = loss + torch.relu((margin + eps) - gap) ** 2
    loss = loss + torch.relu((margin + eps) - py) ** 2
    loss = loss + torch.relu((margin + eps) - xoff) ** 2
    loss = loss + torch.relu((margin + eps) - t) ** 2

    return loss


def loss_bottom_vibration_tsample(
    model,
    mesh,
    t_batch: torch.Tensor,
    component: str = "v",
):
    """
    DC-removed RMS of bottom motion, time-sampled.

    t_batch : (N_t,) or any shape
    component : "u" or "v"
    """
    device = next(model.parameters()).device
    t_batch = t_batch.to(device)

    if mesh.bottom_ids.numel() == 0:
        return torch.zeros((), dtype=torch.float32, device=device)

    bottom_ids = mesh.bottom_ids.to(device, dtype=torch.long)
    comp_idx = 0 if component == "u" else 1

    disp, _, _ = model.eval_with_derivs(t_batch)   # (*T_shape, N_verts, 2)
    disp = disp.reshape(-1, disp.shape[-2], 2)     # (N_t_flat, N_verts, 2)

    bottom_disp = disp[:, bottom_ids, comp_idx]    # (N_t_flat, N_bottom)
    bottom_mean = bottom_disp.mean(dim=0, keepdim=True)
    bottom_centered = bottom_disp - bottom_mean

    rms_per_node = torch.sqrt(torch.mean(bottom_centered**2, dim=0))  # (N_bottom,)
    loss = rms_per_node.mean()
    return loss
