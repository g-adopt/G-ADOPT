# This test is a later addition to the analytical comparisons, as it includes a free surface.
# For a free surface simulation, if the simulation reaches a steady state, then it must be
# that the radial velocity at the surface is zero (following the kinematic bc). If that's the
# case then there is a unique Stokes solution with zero radial (i.e. no normal) flow, and zero
# tangential stress - given by the assess benchmark with the same forcing rhs. In a free surface
# simulation, the actual bc that is being imposed is the normal stress being equal to the free
# surface load (which is just eta scaled by g and rho which are all one here), and therefore the
# free surface steady state solution should be the same as the normal stress derived from the
# analytical free-slip case. We should probably at some point derive the actual time-dependent
# solution here (similar to the Zhong solutions in a box) - and keep in mind that the current
# test doesn't really test the temporal evolution of the free surface dynamics. Hopefully, it's
# a good test to have nonetheless!

# Also as noted below to get the test to converge to the Assess solutions we need to set the variable
# density flag to false. This is probably because the solution is not accounting for buoyancy changes
# to affect the height that the free surface relaxes too. With the flag set to True there is an assymetry
# in the free surface highs (at hot spots) and free surface lows (cold spots), that does not
# occur in the Assess steady state solution

from gadopt import *
from gadopt.utility import CombinedSurfaceMeasure
import numpy
import assess

# Quadrature degree:
_dx = dx(degree=6)
# Projection solver parameters for nullspaces:
_project_solver_parameters = {
    "snes_type": "ksponly",
    "ksp_type": "gmres",
    "pc_type": "sor",
    "mat_type": "aij",
    "ksp_rtol": 1e-12,
}


def model(level, k, nn, do_write=False):
    """The smooth initial condition, cylindrical domain, free-slip boundary condition model.

    Args:
        level: refinement level
        k: radial degree
        nn: wave number per radial degree
        do_write: whether to output the velocity/pressure fields
    """

    rmin, rmax = 1.22, 2.22
    nn = k * nn

    k = Constant(k)
    nn = Constant(nn)

    ncells = level * 128
    nlayers = level * 8

    # Construct a circle mesh and then extrude into a cylinder:
    mesh1d = CircleManifoldMesh(ncells, radius=rmin, degree=2)
    mesh = ExtrudedMesh(mesh1d, layers=nlayers, extrusion_type="radial")
    mesh.cartesian = False
    boundary = get_boundary_ids(mesh)

    # Define geometric quantities
    X = SpatialCoordinate(mesh)
    r = sqrt(X[0]**2 + X[1]**2)
    phi = atan2(X[1], X[0])

    # Set up function spaces - currently using the P2P1 element pair :
    V = VectorFunctionSpace(mesh, "CG", 2)  # velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # pressure function space (scalar)
    Wvec = VectorFunctionSpace(mesh, "CG", 1)  # vector version of W, used for coordinates in anal. pressure solution
    Z = MixedFunctionSpace([V, W, W])  # Mixed function space.
    v, w, w1 = TestFunctions(Z)

    # Nullspaces and near-nullspaces:
    Z_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True)
    Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1])

    # Set up fields on these function spaces - split into each component so that they are easily accessible:
    z = Function(Z)  # a field over the mixed function space Z.
    u, p, eta = split(z)
    u_, p_, eta_ = z.subfunctions

    # Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
    mu = Constant(1.0)  # Constant viscosity

    T = -r**k/rmax**k*cos(nn*phi)  # RHS

    # To get the test to converge to the Assess solutions we need to set the variable density
    # flag to false. This is probably because the solution is not accounting for buoyancy changes
    # to affect the height that the free surface relaxes too. With the flag set to True there is an assymetry
    # in the free surface highs (at hot spots) and free surface lows (cold spots), that does not
    # occur in the Assess steady state solution
    approximation = BoussinesqApproximation(1)
    stokes_bcs = {
        boundary.bottom: {"un": 0},
        # Apply the free surface boundary condition
        boundary.top: {"free_surface": {"variable_rho_fs": False}},
    }

    rho0 = approximation.rho
    g = approximation.g
    tau0 = Constant(2 * nn * mu / (rho0 * g))  # Characteristic time scale (dimensionless)
    log("tau0", tau0)
    dt = Constant(tau0)  # timestep (dimensionless)
    log("dt (dimensionless)", dt)

    stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs, free_surface_dt=dt,
                                 nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                                 near_nullspace=Z_near_nullspace)

    # use tighter tolerances than default to ensure convergence:
    stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-13
    stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-11

    time = Constant(0.0)
    max_timesteps = round(20*tau0/dt)  # Simulation runs for 10 characteristic time scales so end state is close to being fully relaxed
    log("max_timesteps", max_timesteps)

    # Solve system - configured for solving non-linear systems, where everything is on the LHS (as above)
    # and the RHS == 0.

    # Modify ds to work with extruded mesh
    ds = CombinedSurfaceMeasure(mesh, degree=6)

    if do_write:
        # Write output files in VTK format:
        u_.rename("Velocity")
        p_.rename("Pressure")
        eta_.rename("Eta")
        u_file = VTKFile("fs_velocity_{}.pvd".format(level))
        p_file = VTKFile("fs_pressure_{}.pvd".format(level))
        eta_file = VTKFile("fs_eta_{}_theta.pvd".format(level))
        temp_file = VTKFile("fs_temp_{}.pvd".format(level))
        temp_file.write(Function(W, name="T").interpolate(T), Function(W, name="Density").interpolate(approximation.rho_field(0, T)))

    solution = assess.CylindricalStokesSolutionSmoothFreeSlip(int(float(nn)), int(float(k)), nu=float(mu))

    # compute u analytical and error
    uxy = interpolate(as_vector((X[0], X[1])), V)
    u_anal = Function(V, name="AnalyticalVelocity")
    u_anal.dat.data[:] = [solution.velocity_cartesian(xyi) for xyi in uxy.dat.data]

    # compute p analytical and error
    pxy = interpolate(as_vector((X[0], X[1])), Wvec)
    p_anal = Function(W, name="AnalyticalPressure")
    p_anal.dat.data[:] = [solution.pressure_cartesian(xyi) for xyi in pxy.dat.data]

    # compute eta analytical and error
    etaxy = interpolate(as_vector((X[0], X[1])), Wvec)
    eta_anal = Function(W, name="AnalyticalEta")
    eta_anal.dat.data[:] = [-solution.radial_stress_cartesian(xyi) for xyi in etaxy.dat.data]

    steady_state_tolerance = 1e-9
    u_old = Function(V, name="VelocityOld")

    u_error = Function(V, name="VelocityError").assign(u_-u_anal)
    p_error = Function(W, name="PressureError").assign(p_-p_anal)
    eta_error = Function(W, name="EtaError").assign(eta_-eta_anal)

    # Now perform the time loop:
    for timestep in range(1, 20):

        # Solve Stokes sytem:
        stokes_solver.solve()
        time.assign(time + dt)

        # Calculate L2-norm of change in velocity:
        maxchange = sqrt(assemble((u - u_old)**2 * dx))
        log("maxchange = ", maxchange)

        u_old.assign(u_)

        if do_write:
            u_error.assign(u_-u_anal)
            p_error.assign(p_-p_anal)
            eta_error.assign(eta_-eta_anal)

            # Write output:
            u_file.write(u_, u_anal, u_error)
            p_file.write(p_, p_anal, p_error)
            eta_file.write(eta_, eta_anal, eta_error)

        if maxchange < steady_state_tolerance:
            log("Steady-state achieved -- exiting time-step loop")
            break

    u_error.assign(u_-u_anal)
    p_error.assign(p_-p_anal)
    eta_error.assign(eta_-eta_anal)

    l2anal_u = numpy.sqrt(assemble(dot(u_anal, u_anal)*_dx))
    l2anal_p = numpy.sqrt(assemble(dot(p_anal, p_anal)*_dx))
    l2anal_eta = numpy.sqrt(assemble(dot(eta_anal, eta_anal)*ds(top_id)))
    l2error_u = numpy.sqrt(assemble(dot(u_error, u_error)*_dx))
    l2error_p = numpy.sqrt(assemble(dot(p_error, p_error)*_dx))
    l2error_eta = numpy.sqrt(assemble(dot(eta_error, eta_error)*ds(top_id)))

    return l2error_u, l2error_p, l2error_eta, l2anal_u, l2anal_p, l2anal_eta
