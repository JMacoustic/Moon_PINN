import torch
from models.fourier_galerkin.residuals import *
from models.fourier_galerkin.pinn import *
from utils.utils import thickness_from_constraint


def evaluate_geometry_objective(
    px,
    py,
    xoff,
    design,
    mesh,
    model,
    train_cfg,
    material,
    device,
):
    """
    Evaluate scalar geometry objective J for given (px, py, xoff).

    Uses:
      - energy_loss_fourier    (if geom_use_pde)
      - loss_boundary_sine     (if geom_use_bc)
      - loss_initial_condition (if geom_use_ic)
      - loss_bottom_vibration  (if geom_use_vib)

    Geometry is temporarily updated and then restored.
    """

    # ---- cache current geom ----
    px_c, py_c = mesh.pitch
    x_c = mesh.x_offset
    t_c = mesh.thickness

    # ---- set temp geometry ----
    t_new = thickness_from_constraint(design.C, px, py, xoff)
    mesh.adjust_geometry((px, py), xoff, t_new)
    model.set_mesh(mesh)

    # ---- rebuild FE matrices for this geometry (for energy term) ----
    M, Kmat = build_cst_mk(
        verts=mesh.verts_torch,
        tris=mesh.tris_torch,
        E=material.E,
        nu=material.nu,
        rho=material.rho,
        th=material.z_width,
    )
    M    = M.to(device)
    Kmat = Kmat.to(device)

    # external nodal forces (same logic as in train_cst_pinn)
    f_verts = torch.zeros_like(mesh.verts_torch)
    P0 = train_cfg.payload_P0
    if mesh.bottom_ids.numel() > 0:
        f_verts[mesh.bottom_ids, 1] = P0
    f_verts = f_verts.to(device)

    # time samples
    t_batch = torch.linspace(0.0, train_cfg.T, steps=train_cfg.time_steps, device=device)

    # initial condition target
    u0_verts = torch.zeros_like(mesh.verts_torch).to(device)

    # top BC ids
    bc_ids = mesh.top_ids.to(device, dtype=torch.long)

    # ---- compute objective terms (no geom grad needed, so no autograd for design) ----
    use_field_terms = (train_cfg.geom_use_pde or train_cfg.geom_use_bc or train_cfg.geom_use_ic)
    use_vib_term    = train_cfg.geom_use_vib

    cm = torch.enable_grad() if use_field_terms else torch.no_grad()
    with cm:
        J = 0.0

        # field-based terms: energy (PDE), BC, IC
        if use_field_terms:
            if train_cfg.geom_use_pde:
                L_energy = energy_loss_fourier(t_batch, model, M, Kmat, f_verts, C=None)
            else:
                L_energy = torch.tensor(0.0, device=device)

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
                L_bc_top = torch.tensor(0.0, device=device)

            if train_cfg.geom_use_ic:
                L_ic = loss_initial_condition(model, u0_verts)
            else:
                L_ic = torch.tensor(0.0, device=device)

            J_field = (
                train_cfg.w_pde      * L_energy
                + train_cfg.w_bc_top * L_bc_top
                + train_cfg.w_ic     * L_ic
            )

            J += float(J_field)

        # vibration term: bottom RMS
        if use_vib_term:
            L_vib = loss_bottom_vibration(
                model,
                mesh,
                T=train_cfg.T,
                time_steps=train_cfg.time_steps,
                component="v",   # or "uv" if you want both
            )
            J += float(train_cfg.w_vib * L_vib)

    # ---- restore original geometry ----
    mesh.adjust_geometry((px_c, py_c), x_c, t_c)
    if hasattr(model, "set_mesh"):
        model.set_mesh(mesh)

    return J