# # Rayleigh-Taylor instability
# ---

# Rationale
# -

# One may wish to simulate a geodynamical flow involving multiple physical phases. A
# possible approach is to approximate phases as immiscible, forming a single fluid that
# still behaves as a single-phase Stokes flow. Under this approximation, it is usual to
# refer to immiscible phases as materials and the resulting simulations as
# multi-material. In such simulations, each material occupies part of the numerical
# domain and has intrinsic physical properties, such as density and viscosity. Across a
# material interface, physical properties transition from one value to another, either
# sharply or smoothly. In the former case, physical property fields exhibit
# discontinuities, whilst in the latter case, an averaging scheme weights contributions
# from each material.

# Numerical approach
# -

# We employ an interface-capturing approach called the conservative level-set method
# ([Olsson and Kreiss, 2005](https://doi.org/10.1016/j.jcp.2005.04.007)) to model the
# coexistence of multiple materials in the numerical domain. Level-set methods associate
# each material interface to a mathematical field measuring the distance from that
# interface. In the conservative level-set approach, the classic signed-distance
# function, $\phi$, employed in the level-set method is transformed into a smooth step
# function, $\psi$:

# $$\psi(\mathbf{x}, t) = \frac{1}{2} \left[
# \mathrm{tanh} \left( \frac{\phi(\mathbf{x}, t)}{2\epsilon} \right) + 1
# \right]$$

# Throughout the simulation, the level-set field is advected with the flow:

# $$\frac{\partial \psi}{\partial t} + \nabla \cdot \left( \mathbf{u}\psi \right) = 0$$

# Advection of the level set modifies the shape of the initial profile. In other words,
# the signed-distance property underpinning the smooth step function is lost. To
# maintain the original profile whilst the simulation proceeds, a reinitialisation
# procedure is employed. We choose the equation proposed in [Parameswaran and Mandal
# (2023)](https://doi.org/10.1016/j.euromechflu.2022.11.001):

# $$\frac{\partial \psi}{\partial \tau_{n}} = \theta \left[
# -\psi \left( 1 - \psi \right) \left( 1 - 2\psi \right)
# + \epsilon \left( 1 - 2\psi \right) \lvert\nabla\psi\rvert
# \right]$$

# This example
# -

# Here, we consider the isoviscous Rayleigh-Taylor instability presented in [van Keken
# et al. (1997)](https://doi.org/10.1029/97JB01353). Inside a 2-D domain, a buoyant,
# lighter material sits beneath a denser material. The initial material interface
# promotes the development of a rising instability on the domain's left-hand side, and
# further convective dynamics occur throughout the remainder of the simulation. We
# describe below the implementation of this problem using G-ADOPT.

# As with all examples, the first step is to import the `gadopt` package, which also
# provides access to Firedrake and associated functionality.

from gadopt import *

# We next set up the mesh and function spaces and specify functions to hold our
# solutions, as in our previous tutorials.

# +
nx, ny = 64, 64  # Number of cells in x and y directions
lx, ly = 0.9142, 1  # Domain dimensions in x and y directions
# Rectangle mesh generated via Firedrake
mesh = RectangleMesh(nx, ny, lx, ly, quadrilateral=True)
mesh.cartesian = True  # Tag the mesh as Cartesian to inform other G-ADOPT objects.
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

V = VectorFunctionSpace(mesh, "Q", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "Q", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])  # Stokes function space (mixed)
Q = FunctionSpace(mesh, "Q", 2)  # Temperature function space (scalar)
K = FunctionSpace(mesh, "DQ", 2)  # Level-set function space (scalar, discontinuous)
R = FunctionSpace(mesh, "R", 0)  # Real space (constants across the domain)

z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Indexed expressions for velocity and pressure
z.subfunctions[0].rename("Velocity")  # Associated Firedrake function for velocity
z.subfunctions[1].rename("Pressure")  # Associated Firedrake function for pressure
T = Function(Q, name="Temperature")  # Firedrake function for temperature
psi = Function(K, name="Level set")  # Firedrake function for level set
# -

# We now initialise the level-set field. All we have to provide to G-ADOPT is a
# mathematical description of the interface location and use the available API. In this
# case, the interface can be represented as a cosine, requiring a callable to be
# provided. Under the hood, G-ADOPT uses the `Shapely` library to determine the
# signed-distance function associated with the interface. We use G-ADOPT's default
# strategy to obtain a smooth step function profile from the signed-distance function.

# +
from numpy import linspace  # noqa: E402

# Initialise the level-set field. First, determine the signed-distance function at each
# level-set node using a mathematical description of the material-interface location.
# Then, define the thickness of the hyperbolic tangent profile used in the conservative
# level-set approach. Finally, overwrite level-set data array.
interface_coords_x = linspace(0, lx, 1000)
callable_args = (
    interface_deflection := 0.02,
    perturbation_wavelength := 2 * lx,
    initial_interface_y := 0.2,
)
signed_distance_array = signed_distance(
    psi,
    interface_geometry="curve",
    interface_callable="cosine",
    interface_args=(interface_coords_x, *callable_args),
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
Ra = 0  # Thermal Rayleigh number
Ra_c = material_field(psi, [Ra_c_buoyant := 0, Ra_c_dense := 1], interface="sharp")
approximation = BoussinesqApproximation(Ra, Ra_c=Ra_c)
# -

# Let us now verify that the material fields have been correctly initialised. We plot
# the compositional Rayleigh number across the domain.

# + tags=["active-ipynb"]
# fig, axes = plt.subplots()
# axes.set_aspect("equal")
# contours = tricontourf(
#     Function(psi).interpolate(Ra_c), levels=linspace(0, 1, 11), axes=axes
# )
# fig.colorbar(contours, label="Compositional Rayleigh number")
# -

# As with the previous examples, we set up an instance of the `TimestepAdaptor` class
# for controlling the time-step length (via a CFL criterion) whilst the simulation
# advances in time. We specify the initial time, initial time step $\Delta t$, and
# output frequency (in time units).

time_now = 0  # Initial time
delta_t = Function(R).assign(1.0)  # Initial time step
output_frequency = 10  # Frequency (based on simulation time) at which to output
t_adapt = TimestepAdaptor(
    delta_t, u, V, target_cfl=0.6, maximum_timestep=output_frequency
)  # Current level-set advection requires a CFL condition that should not exceed 0.6.

# Here, we set up the variational problem for the Stokes and level-set systems. The
# former depends on the approximation defined above, and the latter includes both
# advection and reinitialisation components.

# +
# This problem setup has a constant pressure nullspace, which corresponds to the default
# case handled in G-ADOPT.
Z_nullspace = create_stokes_nullspace(Z)

# Boundary conditions for the Stokes system: no slip at the top and bottom and free slip
# on the left and right sides.
stokes_bcs = {
    bottom_id: {"u": 0},
    top_id: {"u": 0},
    left_id: {"ux": 0},
    right_id: {"ux": 0},
}
# Instantiate a solver object for the Stokes system and perform a solve to obtain
# initial pressure and velocity.
stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
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
output_file.write(*z.subfunctions, psi, time=time_now)

plog = ParameterLog("params.log", mesh)
plog.log_str("step time dt u_rms entrainment")

gd = GeodynamicalDiagnostics(z, T, bottom_id, top_id)

material_area = initial_interface_y * lx  # Area of tracked material in the domain
entrainment_height = 0.2  # Height above which entrainment diagnostic is calculated
# -

# Finally, we initiate the time loop, which runs until the simulation end time is
# attained.

# +
step = 0  # A counter to keep track of loop iterations
output_counter = 1  # A counter to keep track of outputting
time_end = 2000
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

    # Calculate proportion of material entrained above a given height
    buoy_entr = entrainment(psi, material_area, entrainment_height)
    # Log diagnostics
    plog.log_str(f"{step} {time_now} {float(delta_t)} {gd.u_rms()} {buoy_entr}")

    # Write output
    if time_now >= output_counter * output_frequency:
        output_file.write(*z.subfunctions, psi, time=time_now)
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
