# Runs Spiegelman et al. (2016) cases at eta1=1e23,1e24, and 5e24 Pa s and, resp., U0=2.5,5 and 12.5 mm/yr
# at specified nx, ny resolution with variable number of initial Picard iterations and with or without
# stabilisation of the Jacobian as advocated in Fraters et al. (2019)
from gadopt import *
from firedrake.petsc import PETSc
import firedrake
import os.path
import sys
PETSc.Sys.popErrorHandler()

# Quadrature degree:
dx = dx(degree=6)

# Set up geometry:
nx, ny = 64, 16

H = 30e3
year = 86400*365
U0 = Constant(0.0125/year)
mu0 = 1e22
mu1 = Constant(5e24/mu0)  # upper layer background visc.
mu2 = Constant(1e21/mu0)  # lower layer visc. (isovisc.)

H1 = 0.75
r, h, ww = 0.02, 1/12, 1/6
log = PETSc.Sys.Print


def spiegelman(U0, mu1, nx, ny, picard_iterations, stabilisation=False):
    output_dir = f"spiegelman_{float(U0*year)}_{float(mu1*mu0)}_{nx}_{ny}_{picard_iterations}_{stabilisation}"
    log('\n')
    log('\n')
    log('WRITING TO:', output_dir)
    mesh = RectangleMesh(nx, ny, 4, 1, quadrilateral=True)  # Square mesh generated via firedrake
    mesh.cartesian = True
    left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # noqa: F841 Boundary IDs

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.
    # Test functions and functions to hold solutions:
    v, w = TestFunctions(Z)
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    # Functions to interpolate expression into for output:
    mu_f = Function(W, name="Viscosity")
    epsii_f = Function(W, name="StrainRateSecondInvariant")
    alpha_SPD_f = Function(W, name="alpha_SPD")

    # z_nl is used in the Picard linearisation
    z_nl = Function(Z)
    u_nl, p_nl = split(z_nl)

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

    X = SpatialCoordinate(mesh)
    x = X[0]
    d = 1 - X[1]  # depth: d=0 at top and d=1 at bottom

    # solver dicationaries:

    mumps_solver_parameters = {
        "mat_type": "aij",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
    iterative_solver_parameters = {
        "mat_type": "matfree",
        "ksp_type": "fgmres",
        "ksp_monitor": None,
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur",
        "pc_fieldsplit_schur_type": "upper",
        "fieldsplit_0": {
            "ksp_type": "cg",
            "pc_type": "python",
            "pc_python_type": "firedrake.AssembledPC",
            "assembled_pc_type": "gamg",
            "assembled_pc_gamg_threshold": 0.01,
            "assembled_pc_gamg_square_graph": 100,
            "ksp_rtol": 1e-10,
            "ksp_converged_reason": None,
        },
        "fieldsplit_1": {
            "ksp_type": "preonly",
            "ksp_converged_reason": None,
            "pc_type": "python",
            "pc_python_type": "firedrake.MassInvPC",
            "Mp_pc_type": "ksp",
            "Mp_ksp_ksp_type": "cg",
            "Mp_ksp_pc_type": "sor",
        }
    }

    # initial Picard solve, extra tight tolerance
    initial_picard_solver_parameters = {
        "snes_type": "ksponly",
        "snes_monitor": None
    }
    initial_picard_solver_parameters.update(iterative_solver_parameters)
    initial_picard_solver_parameters['ksp_rtol'] = 1e-16

    picard_solver_parameters = {
        "snes_type": "ksponly",
        "snes_monitor": None,
    }
    picard_solver_parameters.update(mumps_solver_parameters)

    newton_solver_parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "l2",
        "snes_max_it": 50,
        "snes_atol": 0,
        "snes_rtol": 1e-16,
        "snes_stol": 0,
        "snes_monitor": ":"+os.path.join(output_dir, "newton.txt"),
        "snes_converged_reason": None,
    }
    newton_solver_parameters.update(mumps_solver_parameters)

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    g = Constant(9.81)
    rho0 = Constant(2700)
    plith = g*rho0*d*H / (U0*mu0/H)
    phi = Constant(30/180*pi)  # friction angle
    C = Constant(1e8/(mu0*U0/H))  # Cohesion coeffficient
    # Drucker-Prager:
    A = Constant(C*cos(phi))
    B = Constant(sin(phi))
    alpha = Constant(1)

    # switch = 1: use full viscoplastic viscosity
    # switch = 0: just linear viscosity using just eta1 and eta2 in the 2 areas
    # the latter option is used for the initial Picard solve, as the
    # algebraic viscoplastic formula is not defined for u=epsii=0
    # (it is of course defined in the limit u->0 but then mu -> mu1)
    switch = Constant(1.)

    # interface between eta1 and eta2 with rounded corner (although we're
    # completely underresolved for that) as described in Spiegelman
    dx0 = abs(x-2)
    interface_depth = H1 - conditional(dx0 < ww/2, h, 0)
    dx1 = ww/2 + r - dx0
    interface_depth += conditional(And(dx1 > 0, dx1 < r), sqrt(r**2-dx1**2)-r, 0)
    dx2 = dx0 - (ww/2 - r)
    interface_depth -= conditional(And(dx2 > 0, dx2 < r), sqrt(r**2-dx2**2)-r, 0)

    # write viscosity as function of epsii, so we can reuse it in
    # the Jacobian "stabilisation" term
    def eta(epsii, p):
        mu_plast = (A + B*(plith + alpha*p))/(2*epsii)
        mu1_eff = 1/(1/mu1 + 1/mu_plast)
        mu1_eff = conditional(switch > 0.5, mu1*mu_plast/(mu1+mu_plast), mu1)
        mu1_eff = max_value(mu1_eff, mu2)
        return conditional(d < interface_depth, mu1_eff, mu2)

    # viscosity expression in terms of u and p
    eps = sym(grad(u))
    epsii = sqrt(0.5*inner(eps, eps))
    mu = eta(epsii, p)

    # same expression in terms of u_nl and p_nl used in Picard solve
    eps_nl = sym(grad(u_nl))
    epsii_nl = sqrt(0.5*inner(eps_nl, eps_nl))
    mu_nl = eta(epsii_nl, p_nl)

    # Write output files in VTK format:
    u_, p_ = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
    # Next rename for output:
    u_.rename("Velocity")
    p_.rename("Pressure")
    # Create output file and select output_frequency:
    output_file = VTKFile(os.path.join(output_dir, "output.pvd"))
    picard_file = VTKFile(os.path.join(output_dir, 'picard.pvd'))

    # Construct the Jacobian stabilisation form as in Fraters '19
    #
    # we don't actually use eps2 as a function, but use it as a symbolic coefficient
    # so that we can express the derivative of eta(eps2) wrt eps2
    eps2 = Function(W)
    # a, b as in Fraters: a is simply eps
    a = eps
    # b is the derivative of eta wrt eps, which we evaluate as:
    # b = deta/depsii * depsii/deps
    # we evaluate deta/depsii using eps2 as the symbol for epsii to take
    #    derivative with respect to, and then substitute eps2=epsii
    # depsii/deps = eps/(2*epsii)
    # the Constant(1) is the perturbation: derivative computes a Gateaux derivative for a given perturbation
    b = replace(derivative(eta(eps2, p), eps2, Constant(1)) * eps / (2*eps2), {eps2: epsii})
    abnorm = sqrt(inner(a, a)*inner(b, b))
    beta = (1-inner(b, a)/abnorm)**2*abnorm
    c_safety = Constant(1.0)
    alpha_SPD = conditional(beta < c_safety*2*mu, 1, c_safety*2*mu/beta)

    # SCK:
    T = 0
    approximation_nl = BoussinesqApproximation(0, mu=mu_nl)
    approximation = BoussinesqApproximation(0, mu=mu)
    bcs = {left_id: {'ux': 1}, right_id: {'ux': -1}, bottom_id: {'uy': 0}}
    picard_solver = StokesSolver(z, T, approximation_nl, bcs=bcs,
                                 solver_parameters=initial_picard_solver_parameters)
    newton_solver = StokesSolver(z, T, approximation, bcs=bcs,
                                 solver_parameters=newton_solver_parameters)

    if stabilisation:
        jac = derivative(newton_solver.F, z)
        test, trial = jac.arguments()
        # need a trial function to express the Jacobian as a UFL 2-form:
        u_trial, p_trial = split(trial)
        # the normal full Jacobian used in the Newton iteration is derivative(F_stokes_nl, z)
        # which returns a UFL 2-form where the perturbation is a trial function in Z
        # This already contains the deta/du = deta/deps deps/du term which we want to be able to switch off using alpha_SPD:
        #    alpha_SPD=0   deta/du not included
        #    alpha_SPD=1   full deta/du term included
        # Thus we simply subtract (1-alpha_SPD) times that term
        uvar = ufl.variable(u)
        dmu_du = diff(replace(mu, {u: uvar}), uvar)
        jac -= (1-alpha_SPD) * inner(2*dot(dmu_du, u_trial)*sym(grad(u)), grad(v)) * dx
        newton_solver.J = jac

    # switch off viscoplasticity in initial Picard solve as we start from u=0
    switch.assign(0.)

    picard_solver.solve()
    z_nl.assign(z)

    # this defines Picard iteration 0
    mu_f.interpolate(mu)
    epsii_f.interpolate(epsii)
    picard_file.write(u_, p_, mu_f, epsii_f)

    # output file with Picard residuals
    f_picard = open(os.path.join(output_dir, 'picard.txt'), 'w')
    # switch full viscoplastic rheology back on
    switch.assign(1.)

    # initial solve is done iteratively with extra tight tolerance
    # subsequent iterative solves are done with direct solvers
    picard_solver.solver_parameters = picard_solver_parameters
    picard_solver.setup_solver()

    for i in range(picard_iterations):
        f_picard.write(f"{i:02}: {assemble(picard_solver.F, bcs=picard_solver.strong_bcs, zero_bc_nodes=True).dat.norm}\n")
        picard_solver.solve()
        z_nl.assign(z)
        mu_f.interpolate(mu)
        epsii_f.interpolate(epsii)
        picard_file.write(u_, p_, mu_f, epsii_f)
    f_picard.close()

    try:
        newton_solver.solve()
    except firedrake.exceptions.ConvergenceError:
        # ignore_solver failures, so we still get final output
        # Most of the cases will not actually converge to the specified
        # very tight tolerances, so we always perform the max. n/o iterations
        pass

    mu_f.interpolate(mu)
    epsii_f.interpolate(epsii)
    alpha_SPD_f.interpolate(alpha_SPD)
    output_file.write(u_, p_, mu_f, epsii_f, alpha_SPD_f)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        ui, mui, nx, ny, picard_iterations, stab = sys.argv[1].split("_")
        spiegelman(
            Constant(float(ui) / year),
            Constant(float(mui) / mu0),
            int(nx),
            int(ny),
            int(picard_iterations),
            stabilisation=stab == "True",
        )

    else:
        for ui, mui in zip([2.5e-3, 5e-3, 12.5e-3], [1e23, 1e24, 5e24]):
            # 50 initial Picard iterations is used in the graph (which cuts off after 50 iterations)
            # to obtain the converge of a pure Picard solve, i.e. the subsequent Newton solve is
            # ignored in that case
            for picard_iterations in [50, 0, 5, 15, 25]:
                for stab in [False, True]:
                    spiegelman(Constant(ui/year), Constant(mui/mu0), nx, ny, picard_iterations, stabilisation=stab)
