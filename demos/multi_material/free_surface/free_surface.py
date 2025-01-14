# Spontaneous subduction
# ======================

# This example
# ------------

# Here, we consider the spontaneous subduction benchmark presented in [Schmeling et al.
# (2008)](https://doi.org/10.1016/j.pepi.2008.06.028).
# Inside a 2-D domain, a lithosphere sinks into the upper mantle under its own negative
# buoyancy. Here we consider the scenario where the top boundary is modelled as a free
# surface. We describe below how to implement this problem using G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
nx, ny = 256, 64  # Number of cells in x and y directions
lx, ly = 3e6, 7e5  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects.
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "Q", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W, W])  # Stokes function space (mixed)
Q = FunctionSpace(mesh, "Q", 2)  # Temperature function space (scalar)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space (constants across the domain)

z = Function(Z)  # A field over the mixed function space Z
u, p, eta = split(z)  # Indexed expressions for velocity, pressure, and free surface
z.subfunctions[0].rename("Velocity")  # Associated Firedrake function for velocity
z.subfunctions[1].rename("Pressure")  # Associated Firedrake function for pressure
z.subfunctions[2].rename("Free surface")  # Associated Firedrake function for surface
T = Function(Q, name="Temperature")  # Firedrake function for temperature
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now initialise the level-set field. All we have to provide to G-ADOPT is a
# mathematical description of the interface location and use the available API. In this
# case, the interface can be represented as a polygon, requiring vertex coordinates to
# be provided. Under the hood, G-ADOPT uses the `Shapely` library to determine the
# signed-distance function associated with the interface. We use G-ADOPT's default
# strategy to obtain a smooth step function profile from the signed-distance function.

# +
# Initialise the level-set field. First, determine the signed-distance function at each
# level-set node using a mathematical description of the material-interface location.
# Then, define the thickness of the hyperbolic tangent profile used in the conservative
# level-set approach. Finally, overwrite level-set data array.
interface_coords = [
    (lx, ly),
    (1e6, ly),
    (1e6, 5e5),
    (1.1e6, 5e5),
    (1.1e6, 6e5),
    (lx, 6e5),
]
boundary_coords = [(lx, ly)]
signed_distance_array = signed_distance(
    psi,
    interface_geometry="polygon",
    interface_coordinates=interface_coords,
    boundary_coordinates=boundary_coords,
)
epsilon = interface_thickness(psi)
psi.dat.data[:] = conservative_level_set(signed_distance_array, epsilon)
# -

# Let us visualise the location of the material interface that we have just initialised.
# To this end, we use Firedrake's built-in plotting functionality.

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt

# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0, 1, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level-set")
# -

# We next define the material fields and instantiate the approximation. Here, the system
# of equations is non-dimensional and only includes compositional buoyancy under the
# Boussinesq approximation. Moreover, physical parameters are constant through space
# apart from density. As a result, the system is fully defined by the compositional
# Rayleigh number. We use the `material_field` function to define its value throughout
# the domain (including the shape of the material interface transition) and provide it
# to our approximation.

# +
# Material fields defined based on each material value and location
mu = material_field(psi, [mu_mantle := 1e21, mu_slab := 1e23], interface="geometric")
rho_material = material_field(
    psi, [rho_mantle := 3200, rho_slab := 3300], interface="sharp"
)

approximation = BoussinesqApproximation(
    Ra := 1, Ra_c=1, g=9.81, mu=mu, rho=rho_mantle, delta_rho=rho_material - rho_mantle
)
# -

# Let us now verify that the material fields have been correctly initialised. We plot
# the compositional Rayleigh number across the domain.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots(ncols=2)
# for ax in axes:
#     ax.set_aspect("equal")
# contours = tricontourf(
#     Function(psi).interpolate(mu), levels=linspace(1e21, 1e23, 11), axes=axes[0]
# )
# fig.colorbar(contours, ax=axes[0], label="Viscosity (Pa s)")
# contours = tricontourf(
#     Function(psi).interpolate(rho_material),
#     levels=linspace(3200, 3300, 11),
#     axes=axes[1],
# )
# fig.colorbar(contours, ax=axes[1], label="Density (kg/m^3)")
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

myr_to_seconds = 1e6 * 365.25 * 8.64e4
time_now = 0  # Initial time
delta_t = Function(R).assign(1e11)  # Initial time step
# Frequency (based on simulation time) at which to output
output_frequency = 0.8 * myr_to_seconds
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# Here, we set up the variational problem for the Stokes and level-set systems. The
# former depends on the approximation defined above, and the latter includes both
# advection and reinitialisation components. Subcycling is available for level-set
# advection and is mainly useful when the problem at hand involves multiple CFL
# conditions, with the CFL for level-set advection being the most restrictive.
# Boundary conditions are specified next: a free surface at the top and free slip at all
# other boundaries. No boundary conditions are required for level set, as the numerical
# domain is closed.

# +
rho_ext = 0
stokes_bcs = {
    bottom_id: {"uy": 0},
    top_id: {"free_surface": {"delta_rho_fs": rho_mantle - rho_ext}},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
# Instantiate a solver object for the Stokes system and perform a solve to obtain
# initial pressure and velocity.
stokes_solver = StokesSolver(
    z, T, approximation, bcs=stokes_bcs, free_surface_dt=delta_t
)
stokes_solver.solve()

# Instantiate a solver object for level-set advection and reinitialisation. G-ADOPT
# provides default values for most arguments; we only provide those that do not have
# one. No boundary conditions are required, as the numerical domain is closed.
adv_kwargs = {"u": u, "timestep": delta_t}
reini_kwargs = {"epsilon": epsilon}
level_set_solver = LevelSetSolver(psi, adv_kwargs=adv_kwargs, reini_kwargs=reini_kwargs)
# -

# We now set up our output. To do so, we create the output file as a ParaView Data file
# that uses the XML-based VTK file format. We also open a file for logging, instantiate
# G-ADOPT geodynamical diagnostic utility, and define some parameters specific to this
# problem.

# +
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, psi, time=time_now / myr_to_seconds)

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms slab_tip_depth")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id)
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
from gadopt.level_set_tools import min_max_height  # noqa: E402

step = 0  # A counter to keep track of looping
output_counter = 1  # A counter to keep track of outputting
time_end = 60 * myr_to_seconds
while True:
    # Update timestep
    if time_end - time_now < output_frequency:
        t_adapt.maximum_timestep = time_end - time_now
    t_adapt.update_timestep()

    # Advect level set
    level_set_solver.solve()
    # Solve Stokes sytem
    stokes_solver.solve()

    # Increment iteration count and time
    step += 1
    time_now += float(delta_t)

    # Log diagnostics
    plog.log_str(
        f"{step} {time_now} {float(delta_t)} {gd.u_rms()} "
        f"{(ly - min_max_height(psi, epsilon, side=1, mode='min')) / 1e3}"
    )

    # Write output
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, psi, time=time_now / myr_to_seconds)
        output_counter += 1

    # Check if simulation has completed
    if time_now >= time_end:
        plog.close()  # Close logging file

        # Checkpoint solution fields to disk
        with CheckpointFile("Final_State.h5", "w") as final_checkpoint:
            final_checkpoint.save_mesh(mesh)
            final_checkpoint.save_function(z, name="Stokes")
            final_checkpoint.save_function(psi, name="Level set")

        log("Reached end of simulation -- exiting time-step loop")
        break
# -

# Let us finally examine the location of the material interface at the end of the
# simulation.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(psi, levels=linspace(0, 1, 11), axes=axes, cmap="PiYG")
# tricontour(psi, axes=axes, levels=[0.5])
# fig.colorbar(contours, label="Conservative level-set")
# -
