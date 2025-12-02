import torch
from models.fourier_galerkin.residuals import *
from models.fourier_galerkin.pinn import *
from models.fourier_galerkin.multifreq_pinn import *

def evaluate_geometry_objective(
    mesh,
    model,
    train_cfg,
    material,
    device,
    M: torch.Tensor | None = None,
    Kmat: torch.Tensor | None = None,
):
    """
    Evaluate scalar geometry objective J for the *current* mesh parameters.

    Geometry parameters (pitch, x_offset, thickness) are assumed to be
    stored as tensors/Parameters inside `mesh`. This function must be
    fully differentiable w.r.t. those parameters.

    Returns
    -------
    J : torch.Tensor scalar (requires_grad=True iff geometry terms used)
    """

    # Make sure verts_torch matches current pitch/x_offset/thickness
    mesh.adjust_geometry()
    if hasattr(model, "set_mesh"):
        model.set_mesh(mesh)

    # --- FE matrices (optional) ---
    # If you want geometry gradients **through** M, Kmat, do NOT detach!
    if M is None or Kmat is None:
        M, Kmat = build_cst_mk(
            verts=mesh.verts_torch,   # depends on pitch/x_offset
            tris=mesh.tris_torch,
            E=material.E,
            nu=material.nu,
            rho=material.rho,
            th=material.z_width,
        )
        M    = M.to(device)
        Kmat = Kmat.to(device)
    else:
        M    = M.to(device)
        Kmat = Kmat.to(device)

    # --- external nodal forces ---
    f_verts = torch.zeros_like(mesh.verts_torch, device=device)
    if mesh.bottom_ids.numel() > 0:
        P0 = train_cfg.payload_P0
        bottom_ids = mesh.bottom_ids.to(device=device, dtype=torch.long)
        f_verts[mesh.bottom_ids, 1] = P0

    # --- time samples, IC, BC indices ---
    t_batch  = torch.linspace(0.0, train_cfg.T, steps=train_cfg.time_steps, device=device)
    u0_verts = torch.zeros_like(mesh.verts_torch, device=device)
    bc_ids   = mesh.top_ids.to(device, dtype=torch.long)

    # --------------------------------------------------
    # IMPORTANT: NO torch.no_grad(), NO float(.), etc.
    # Everything below must stay in the graph.
    # --------------------------------------------------
    use_field_terms = (train_cfg.geom_use_pde or train_cfg.geom_use_bc or train_cfg.geom_use_ic)
    use_vib_term    = train_cfg.geom_use_vib

    J = torch.zeros((), device=device)

    # --- field-based terms ---
    if use_field_terms:
        if train_cfg.geom_use_pde:
            L_energy = energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None)
        else:
            L_energy = torch.zeros((), device=device)

        if train_cfg.geom_use_bc:
            L_bc_top = loss_boundary_sine(
                model,
                t_batch,
                bc_ids,
                amp_y=train_cfg.V0,
                phase=0.0,
                offset_y=0.0,
                x_fixed=0.0,
            )
        else:
            L_bc_top = torch.zeros((), device=device)

        if train_cfg.geom_use_ic:
            L_ic = loss_initial_condition(model, u0_verts)
        else:
            L_ic = torch.zeros((), device=device)

        J_field = (
            train_cfg.w_pde      * L_energy
            + train_cfg.w_bc_top * L_bc_top
            + train_cfg.w_ic     * L_ic
        )

        J = J + J_field

    # --- vibration term ---
    if use_vib_term:
        L_vib = loss_bottom_vibration(
            model,
            mesh,
            component="v",   # or "uv"
        )
        J = J + train_cfg.w_vib * L_vib

    return J
