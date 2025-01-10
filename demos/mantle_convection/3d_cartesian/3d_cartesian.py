# Idealised 3-D mantle convection problem
# =======================================================
#
# In this tutorial we highlight the ease at which simulations can be examined in different
# dimensions by modifying the 2-D case presented in our first tutorial. We simulate
# a benchmark case that is well-known within the geodynamical community.
#
# This example
# ------------
#
# We examine a low Rayleigh number isoviscous case: specifically Case 1a from Busse et al. (1994).
# The domain is a box of dimensions $1.0079 \times 0.6283 \times 1$. The initial temperature distribution,
# chosen to produce a single ascending and descending flow, at $x = y = 0$ and $(x = 1.0079, y = 0.6283)$,
# respectively, is prescribed as:
#
# $$    T(x,y,z) = \Bigl[ \frac{\mbox{erf}(4(1-z)) + \mbox{erf}(-4z)+1}{2} \Bigr] + A [\cos(\pi x/1.0079) + \cos(\pi y/0.6283)]\sin(\pi z) $$
#
# where $A=0.2$ is the amplitude of the initial perturbation. We note that this initial condition differs to that
# specified in Busse et al. (1994), through the addition of boundary layers at the bottom and top of the domain (through the $\mbox{erf}$ terms),
# although it more consistently drives solutions towards the final published steady-state results. Boundary conditions for
# temperature are T = 0 at the surface (z = 1) and T = 1 at the base (z = 0), with insulating (homogeneous Neumann) sidewalls.
# No‐slip velocity boundary conditions are specified at the top surface and base of the domain, with free‐slip boundary conditions on all
# sidewalls. The Rayleigh number $Ra = 3 \times 10^{4}$.
#
# The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh, function spaces, and specify functions to hold our solutions,
# as with our previous tutorials.
#
# We generate our 3-D mesh by extruding a 2-D quadrilateral
# mesh in the $z$-direction to a layered 3-D hexahedral mesh. Our final mesh has $10 \times 6 \times 10$
# elements, in $x$-, $y$- and $z$-directions, respectively (noting that the default value for layer height is 1 / $nz$).
# For extruded meshes, top and bottom boundaries are tagged by *top* and *bottom*, respectively, whilst boundary
# markers from the base mesh can be used to set boundary conditions on the relevant side of the extruded mesh.
# We note that Firedrake exploits the regularity of extruded meshes to enhance performance.

# +
a, b, c = 1.0079, 0.6283, 1.0
nx, ny, nz = 10, int(b/c * 10), 10
mesh2d = RectangleMesh(nx, ny, a, b, quadrilateral=True)  # Rectangular 2D mesh
mesh = ExtrudedMesh(mesh2d, nz)
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
# -
# We can output function space information, for example the number of degrees
# of freedom (DOF) using log, a utility provided by G-ADOPT.

# + tags=["active-ipynb"]
# log("Number of Velocity DOF:", V.dim())
# log("Number of Pressure DOF:", W.dim())
# log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
# log("Number of Temperature DOF:", Q.dim())
# -

# We next specify the important constants for this problem, and set up the approximation.

Ra = Constant(3e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# As with the previous example, we set up a *Timestep Adaptor*,
# for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time. For the latter,
# we specify the initial time, initial timestep $\Delta t$, and number of
# timesteps. Given the low Rayleigh number, a steady-state tolerance is also specified,
# allowing the simulation to exit when a steady-state has been achieved.

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 20000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)
steady_state_tolerance = 1e-7  # Used to determine if solution has reached a steady state.

# We next set up and initialise our Temperature field (in 3-D).

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate(0.5*(erf((1-X[2])*4)+erf(-X[2]*4)+1) + 0.2*(cos(pi*X[0]/a)+cos(pi*X[1]/b))*sin(pi*X[2]))

# With closed boundaries on all domain boundaries, this problem has a constant pressure nullspace, which is
# handled identically to our 2-D tutorials.

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Aside from differences in the computational geometry and temperature initial condition, the first major change between
# our initial 2-D tutorial and this example occurs in our solution strategy, although most of this is taken care of under the hood by G-ADOPT.
# By default, G-ADOPT uses direct solvers in 2-D. However, in 3-D, we default to iterative solvers. Although the user need not
# alter these, some key information is provided here, for context. Later in the tutorial, we also expose some of these solver
# options to the user, to showcase how our default parameters can be modified.
#
# For the Stokes system, we configure the Schur complement approach as described in Section of 4.3 of Davies et al. (2022),
# using PETSc's fieldsplit preconditioner type, which provides a class of preconditioners for mixed problems that allows a user
# to apply different preconditioners to different blocks of the system. The *fieldsplit\_0* entries configure solver options for the velocity block.
# The linear systems associated with this matrix are solved using a combination of the Conjugate Gradient method and an algebraic multigrid preconditioner (GAMG).
# The *fieldsplit\_1* entries contain solver options for the Schur complement solve itself. For preconditioning, we approximate the Schur complement matrix with
# a mass matrix scaled by viscosity, with the viscosity provided through the optional *mu* keyword argument to Stokes solver (note since viscosity is constant,
# we do not do so for this tutorial). Since this preconditioner step involves an iterative solve, the Krylov method used for the Schur
# complement needs to be of flexible type, and we use FGMRES by default. Finally, the energy solve is performed through a combination of the
# GMRES Krylov method and SSOR preconditioning.
#
# The GAMG preconditioner can make use of so-called *near-nullspace* modes, to improve performance, ensuring that these are accurately represented at the coarser
# multigrid levels. We therefore create a near-nullspace object consisting of three rotational and three translational modes.

Z_near_nullspace = create_stokes_nullspace(Z, closed=False, rotational=True, translations=[0, 1, 2])

# Boundary conditions are next specified, with zero slip conditions specified for top and bottom boundaries, and free-slip
# on all other boundaries. A temperature of 1 is specified at the base and 0 at the domain's surface.

# +
stokes_bcs = {
    boundary.bottom: {'u': 0},
    boundary.top: {'u': 0},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
    boundary.front: {'uy': 0},
    boundary.back: {'uy': 0},
}

temp_bcs = {
    boundary.bottom: {'T': 1.0},
    boundary.top: {'T': 0.0},
}
# -

# We next set up our output, in VTK format, including a file
# that allows us to visualise the reference state.
# We also open a file for logging and calculate our diagnostic outputs.

# +
output_file = VTKFile("output.pvd")
ref_file = VTKFile('reference_state.pvd')
output_frequency = 50

plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")

gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top)
# -

# We can now setup and solve the variational problem, for both the energy and Stokes equations,
# passing in the approximation, nullspace and near-nullspace information configured above.

# +
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             near_nullspace=Z_near_nullspace)
# -

# For all iterative solves, G-ADOPT utilises convergence criterion based on the relative reduction of the
# preconditioned residual, *ksp\_rtol*. These are set to 1e-5 for the *fieldslip\_0* and 1e-4 for *fieldsplit\_1*.
# We can change these default values, by accessing the solver_parameters dictionary, as follows.

stokes_solver.solver_parameters['fieldsplit_0']['ksp_rtol'] = 1e-4
stokes_solver.solver_parameters['fieldsplit_1']['ksp_rtol'] = 1e-3

# We now initiate the time loop, which runs until a steady-state solution has been attained.

for timestep in range(0, timesteps):

    # Write output:
    if timestep % output_frequency == 0:
        output_file.write(*z.subfunctions, T)

    dt = t_adapt.update_timestep()
    time += dt

    # Solve Stokes sytem:
    stokes_solver.solve()

    # Temperature system:
    energy_solver.solve()

    # Compute diagnostics:
    energy_conservation = abs(abs(gd.Nu_top()) - abs(gd.Nu_bottom()))

    # Calculate L2-norm of change in temperature:
    maxchange = sqrt(assemble((T - energy_solver.T_old)**2 * dx))

    # Log diagnostics:
    plog.log_str(f"{timestep} {time} {float(delta_t)} {maxchange} "
                 f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} {gd.Nu_top()} "
                 f"{gd.Nu_bottom()} {energy_conservation} {gd.T_avg()} ")

    # Leave if steady-state has been achieved:
    if maxchange < steady_state_tolerance:
        log("Steady-state achieved -- exiting time-step loop")
        break

# At the end of the simulation, once a steady-state has been achieved, we close our logging file
# and checkpoint steady state temperature and Stokes solution fields to disk. These can later be
# used to restart a simulation, if required.

# +
plog.close()

with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
    final_checkpoint.save_mesh(mesh)
    final_checkpoint.save_function(T, name="Temperature")
    final_checkpoint.save_function(z, name="Stokes")
