# Idealised 2-D mantle convection problem in a square box
# =======================================================
#
# In this tutorial, we examine an idealised 2-D problem in square box.
# The case is identical to our base case demo, but mesh generation is
# done through gmsh rather than Firedrake.
#
# Let's get started! The first step is to import the gadopt module, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We will set up the problem using a bilinear quadrilateral element
# pair (Q2-Q1) for velocity and pressure, with Q2 elements for
# temperature.
#
# We first need a mesh. For this test, we use a gmsh derived square mesh.
# Tag 14 corresponds to the plane $x=0$; 12 to the $x=1$ plane; 11 to the $y=0$ plane;
# and 13 to the $y=1$ plane. We name these `left`, `right`, `bottom` and `top`,
# respectively.
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points
# in the negative z-direction. This attribute is used by gadopt specifically, not
# Firedrake. By contrast, a non-Cartesian geometry is assumed to have gravity
# pointing in the radially inward direction.

nx, ny = 60, 60  # Number of cells in x and y directions.

mesh = Mesh("square.msh", quadrilateral=True)  # Square mesh generated via gmsh
mesh.cartesian = True
boundary = get_boundary_ids(mesh)

# We also need function spaces, which is achieved by associating the
# mesh with the relevant finite element: V , W and Q are symbolic
# variables representing function spaces. They also contain the
# function space’s computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis.

V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the velocity and pressure spaces to form
# a function space for the mixed Stokes problem, Z.

Z = MixedFunctionSpace([V, W])  # Mixed function space.

# We also specify functions to hold our solutions: z in the mixed
# function space, noting that a symbolic representation of the two
# parts – velocity and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of z Velocity and Pressure.

z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")

# We can output function space information, for example the number of degrees
# of freedom (DOF) using log, a utility provided by gadopt.

# + tags=["active-ipynb"]
# log("Number of Velocity DOF:", V.dim())
# log("Number of Pressure DOF:", W.dim())
# log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())
# log("Number of Temperature DOF:", Q.dim())
# -

# The Rayleigh number for this problem is defined. The viscosity and thermal
# diffusivity are left at their default values (both = 1). We note that viscosity
# could also be a Function, if we wanted spatial variation, and will
# return to this in a subsequent notebook.  These Ra is required to
# create an *Approximation* representing the physical
# setup of the problem (options include Boussinesq, Extended
# Boussinesq, Truncated Anelastic Liquid and Anelastic Liquid), and a
# *Timestep Adaptor*, for controlling the time-step length (via a CFL
# criterion) as the simulation advances in time. For the latter,
# we specify the initial time, initial timestep $\Delta t$, and number of
# timesteps. Given the low Ra, a steady-state tolerance is also specified,
# allowing the simulation to exit when a steady-state has been achieved.

# +
Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

time = 0.0  # Initial time
delta_t = Constant(1e-6)  # Initial time-step
timesteps = 20000  # Maximum number of timesteps
t_adapt = TimestepAdaptor(delta_t, u, V, maximum_timestep=0.1, increase_tolerance=1.5)

steady_state_tolerance = 1e-9  # Used to determine if solution has reached a steady state.
# -

# Mantle convection is an initial and boundary-value problem. We
# assume the initial temperature distribution to be prescribed by
#
# $T(x,y) = (1-y) + 0.05\ cos(\pi x)\ sin(\pi y)$
#
# In the following code, we first obtain symbolic expressions for
# coordinates in the physical mesh and subsequently use these to
# initialize the temperature field.  This is where Firedrake
# transforms a symbolic operation into a numerical computation for the
# first time: the `interpolate` method generates C code that evaluates
# this expression at the nodes of $T$.

X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.05*cos(pi*X[0])*sin(pi*X[1])))

# We can visualise the initial temperature field using Firedrake's
# built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -

# With closed boundaries, and no constraint on pressure anywhere in
# the domain, this problem has a constant pressure nullspace and we
# must ensure that our solver removes this space. To do so, we build a
# nullspace object, which will subsequently be passed to the solver,
# and PETSc will seek a solution in the space orthogonal to the
# provided nullspace.  When building the nullspace object, the
# 'closed' keyword handles the constant pressure nullspace, whilst the
# 'rotational' keyword deals with rotational modes, which, for
# example, manifest in an a annulus domain with free slip top and
# bottom boundaries (as we will see in a later tutorial).

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# We next specify strong Dirichlet boundary conditions for velocity and
# temperature. The user must provide the part of the mesh at which
# each boundary condition applies.  Note how boundary conditions have
# the granularity to be applied to the $x$ and $y$ components of the
# velocity field only, if desired.

# +
stokes_bcs = {
    boundary.bottom: {'uy': 0},
    boundary.top: {'uy': 0},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
}

temp_bcs = {
    boundary.bottom: {'T': 1.0},
    boundary.top: {'T': 0.0},
}
# -

# We next set up our output, in VTK format. To do so, we create the output file
# and specify the output_frequency in timesteps.

output_file = VTKFile("output.pvd")
output_frequency = 50

# Next, we open a file for logging diagnostic output and provide the header. We will be outputting
# the timestep number, the time, the timestep size, the L2 norm of the change in temperature between
# consequtive timesteps, the RMS velocity, the RMS velocity at the surface of the domain, the maximum
# x-component of velocity at the domains surface, the surface Nusselt number, the basal Nusselt number,
# the difference between surface and bottom Nusselt numbers (energy_conservation) and the average temperature
# across the domain. These are computed using the GeodynamicalDiagnostics class, which takes in the Stokes (z)
# and temperature functions, alongside bottom and top boundary IDs.

# +
plog = ParameterLog('params.log', mesh)
plog.log_str("timestep time dt maxchange u_rms u_rms_surf ux_max nu_top nu_base energy avg_t")

gd = GeodynamicalDiagnostics(z, T, boundary.bottom, boundary.top)
# -

# We finally come to solving the variational problem, with solver
# objects for the energy and Stokes systems created. For the energy
# system we pass in the solution field T, velocity u, the physical
# approximation, time step, temporal discretisation approach
# (i.e. implicit middle point), and temperature boundary conditions. For the Stokes
# system, we pass in the solution fields z, Temperature, the physical
# approximation, boundary conditions and the nullspace object.
#
# Given that this model is isoviscous, we can speed up the simulation by specifying a
# constant Jacobian (preventing uneccesary matrix re-assembly).
# We note that solution of the two variational problems is undertaken by PETSc.

# +
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace,
                             constant_jacobian=True)
# -

# We can now initiate the time-loop, with the Stokes and energy
# systems solved seperately. These `solve` calls once again convert
# symbolic mathematics into computation. In the time loop, set here to
# run for a maximum of 20000 time-steps, we output in VTK format every 50 timesteps.
# The timestep itself is updated, using the update_timestep function, with diagnostics logged via the log utility
# at every timestep. At the end of each time step, we calculate the L2-norm of
# the change in temperature and, once this drops below the steady_state_tolerance specified above,
# we exit the timeloop.

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
# -

# We can visualise the final temperature field using Firedrake's
# built-in plotting functionality.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# collection = tripcolor(T, axes=axes, cmap='coolwarm')
# fig.colorbar(collection);
# -
