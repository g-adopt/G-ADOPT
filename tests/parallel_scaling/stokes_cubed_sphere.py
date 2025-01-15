from gadopt import *
import scipy.special
import math


def model(ref_level, nlayers, delta_t, steps=None):
    # Set up geometry:
    rmin, rmax = 1.22, 2.22

    # Construct a CubedSphere mesh and then extrude into a sphere:
    mesh2d = CubedSphereMesh(rmin, refinement_level=ref_level, degree=2)
    mesh = ExtrudedMesh(mesh2d, layers=nlayers, extrusion_type='radial')
    mesh.cartesian = False
    bottom_id, top_id = "bottom", "top"

    # Set up function spaces - currently using the bilinear Q2Q1 element pair:
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Z = MixedFunctionSpace([V, W])  # Mixed function space.

    # Test functions and functions to hold solutions:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p = split(z)  # Returns symbolic UFL expression for u and p

    # Output function space information:
    log("Number of Velocity DOF:", V.dim())
    log("Number of Pressure DOF:", W.dim())
    log("Number of Velocity and Pressure DOF:", V.dim() + W.dim())
    log("Number of Temperature DOF:", Q.dim())

    # Timing info:
    stokes_stage = PETSc.Log.Stage("stokes_solve")
    energy_stage = PETSc.Log.Stage("energy_solve")

    # Set up temperature field and initialise:
    T = Function(Q, name="Temperature")
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2 + X[2]**2)
    theta = atan2(X[1], X[0])  # Theta (longitude - different symbol to Zhong)
    phi = atan2(sqrt(X[0]**2+X[1]**2), X[2])  # Phi (co-latitude - different symbol to Zhong)

    conductive_term = rmin*(rmax - r) / (r*(rmax - rmin))
    # evaluate P_lm node-wise using scipy lpmv
    l, m, eps_c, eps_s = 3, 2, 0.01, 0.01
    Plm = Function(Q, name="P_lm")
    cos_phi = interpolate(cos(phi), Q)
    Plm.dat.data[:] = scipy.special.lpmv(m, l, cos_phi.dat.data_ro)
    Plm.assign(Plm*math.sqrt(((2*l+1)*math.factorial(l-m))/(2*math.pi*math.factorial(l+m))))
    if m == 0:
        Plm.assign(Plm/math.sqrt(2))
    T.interpolate(conductive_term +
                  (eps_c*cos(m*theta) + eps_s*sin(m*theta)) * Plm * sin(pi*(r - rmin)/(rmax-rmin)))

    Ra = Constant(7e3)  # Rayleigh number
    mu = exp(4.605170185988092 * (0.5 - T))
    approximation = BoussinesqApproximation(Ra, mu=mu)

    delta_t = Constant(delta_t)  # Initial time-step

    max_timesteps = steps or 20
    time = 0.0

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

    temp_bcs = {
        bottom_id: {'T': 1.0},
        top_id: {'T': 0.0},
    }
    stokes_bcs = {
        bottom_id: {'un': 0},
        top_id: {'un': 0},
    }

    energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
    energy_solver.solver_parameters['ksp_converged_reason'] = None
    energy_solver.solver_parameters['ksp_view'] = None
    energy_solver.solver_parameters['ksp_rtol'] = 1e-7

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    stokes_solver.solver_parameters['fieldsplit_0']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_monitor_true_residual'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_view'] = None
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-7
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_converged_reason'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_view'] = None
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-5

    # Now perform the time loop:
    for timestep in range(0, max_timesteps):
        dt = float(delta_t)
        time += dt

        # Solve Stokes sytem:
        with stokes_stage:
            stokes_solver.solve()

        # Temperature system:
        with energy_stage:
            energy_solver.solve()
