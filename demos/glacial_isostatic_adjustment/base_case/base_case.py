# Idealised 2-D viscoelastic loading problem in a square box
# =======================================================
#
# In this tutorial, we examine an idealised 2-D loading problem in a square box.
# Here we will focus purely on viscoelastic deformation by a surface load, i.e. a synthetic
# ice sheet!
#
# You may already have seen how G-ADOPT can be applied to mantle convection problems in our
# other tutorials. Generally the setup of the G-ADOPT model should be familiar but the equations
# we are solving and the necessary input fields will be slightly different.
#
# Governing equations
# -------------------
# Let's start by reviewing some equations! Similar to mantle convection, the governing equations
# for viscoelastic loading are derived from the conservation laws of mass and momentum.
#
# The conservation of momentum is
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma} - \rho \nabla \Phi = 0,
# \end{equation}
# where $\boldsymbol{\sigma}$ is the full stress tensor, $\rho$ is the density and $\Phi$ is the
# gravitational potential field. As with mantle convection, we have neglected inertial terms.
#
# For incompressible materials conservation of mass is
# \begin{equation}
#     \frac{\partial \rho}{\partial t} + \textbf{v} \cdot \nabla \rho = 0,
# \end{equation}
# where $\textbf{v}$ is the velocity. For the moment we are focusing on incompressible materials where
# \begin{equation}
#     \nabla \cdot \textbf{v} = 0.
# \end{equation}
#
#

# Linearisation
# -------------
# The conservation of momentum is usually linearised due to the small displacements relative to the
# depth of the mantle, i.e.
# \begin{equation}
#     \rho = \rho_0 + \rho_1,
# \end{equation}
# \begin{equation}
#     \boldsymbol{\sigma} = \boldsymbol{\sigma}_0 + \boldsymbol{\sigma}_1,
# \end{equation}
# \begin{equation}
#     \nabla \Phi = \nabla \Phi_0 + \nabla \Phi_1 = \textbf{g} + \nabla \Phi_1.
# \end{equation}
# Subbing this into the momentum equation and neglecting higher order terms gives
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_0 + \nabla \cdot \boldsymbol{\sigma}_1  - \rho_0 \textbf{g}  - \rho_1 \textbf{g} - \rho_0 \nabla \Phi_1 = 0.
# \end{equation}
# The background state is assumed to be in hydrostatic equilibrium so
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_0 = \rho_0 \textbf{g}.
# \end{equation}
# Therefore we can simplify the momentum balance to
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1   - \rho_1 \textbf{g} - \rho_0 \nabla \Phi_1 = 0.
# \end{equation}
# For this tutorial we are going to ignore changes in the gravitational field, so the last term drops out, giving
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1   - \rho_1 \textbf{g}  = 0.
# \end{equation}
#
# Linearising the conservation of mass gives
# \begin{equation}
#     \frac{\partial (\rho_0+\rho_1)}{\partial t} + \textbf{v} \cdot \nabla (\rho_0 + \rho_1) = 0.
# \end{equation}
# After neglecting higher order terms and assuming the background density field, $\rho_0$, does not vary in time,
# the conservation of mass becomes
# \begin{equation}
#     \frac{\partial \rho_1}{\partial t} + \textbf{v} \cdot \nabla \rho_0  = 0.
# \end{equation}
# Integrating this equation with respect to time gives
# \begin{equation}
#     \rho_1 = - \textbf{u} \cdot \nabla \rho_0 ,
# \end{equation}
# assuming $\rho_1 = 0$ at initial times. Note that $\textbf{u}$ is now the displacement vector. The density
# perturbation $\rho_1$ is referred to as the *Eulerian density pertubation* by the GIA community.
#

# Incremental Lagrangian Stress Tensor
# -------------------------------------
# The GIA community usually reformulates the linearised momentum equation in terms of the
# *Incremental lagrangian stress tensor*. This can be traced back to the early roots of GIA
# modellers adopting Laplace transform methods. The idea behind this is to convert the time
# dependent *viscoelastic* problem to a time independent *elastic* problem by the correspondence principle.
#
# From an elastic wave theory point of view, Dahlen and Tromp (1998) make the point that it is the
# Lagrangian perturbation in stress not the Eulerian perturbation that is related to the displacement
# gradient by the elastic parameters. Transforming between the Lagrangian perturbation in stress and
# the Eulerian description is given by
# \begin{equation}
#      \boldsymbol{\sigma}_{L1} =  \boldsymbol{\sigma}_1 + \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ,
# \end{equation}
# where $\boldsymbol{\sigma}_{L1}$ is the Incremental lagrangian stress tensor.
#
# This is effectively accounting for an advection of a background quantity when translating between the Eulerian
# and Lagrangian frames of reference through a first order Taylor series expansion.
#
# This advection of prestress can be important for very long wavelength loads. Cathles (1975) estimates that the
# term becomes leading order when the wavelength is greater than 30000 km for typical Earth parameters, i.e. only
# when the wavelength is the same order of magnitude as the circumference of the Earth.
#
# For the viscoelastic problem, however, this term is crucial because it acts as a restoring force to
# isostatic equilibrium. If the Laplace transform methods do not include this term a load placed on the surface
# of the Earth will keep sinking (Wu and Peltier, 1982)!
#
# Subbing into the stress balance gives
#
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_1 = \nabla \cdot (\boldsymbol{\sigma}_{L1}  - \textbf{u} \cdot \nabla \boldsymbol{\sigma}_0 ) = \nabla \cdot \boldsymbol{\sigma}_{L1} - \nabla (\rho_0 g u_r)
# \end{equation}
# where $u_r$ is the radial displacement vector.
#

# Maxwell Rheology
# ----------------
#
# The GIA community generally model the mantle as an incompressible Maxwell solid. The conceptual picture is a
# spring and a dashpot connected together in series (Ranalli, 1995). For this viscoelastic model the elastic and
# viscous stresses are the same but the total displacements combine.
#
# The viscous constitutive relationship is
# \begin{equation}
#     \overset{\cdot}{\boldsymbol{\epsilon}}^v = \dfrac{1}{2 \eta} (\boldsymbol{\sigma}_{L1} + p \textbf{I}).
# \end{equation}
# The corresponding elastic constitutive equation is
# \begin{equation}
#     \boldsymbol{\epsilon}^e  = \dfrac{1}{2 \mu} (\boldsymbol{\sigma}_{L1} + p \textbf{I})
# \end{equation}
# where $\overset{\cdot}{\boldsymbol{\epsilon}}^v $ is the viscous strain rate tensor, $\boldsymbol{\epsilon}^e$ is the
# elastic strain tensor, $\eta$  is the viscosity, $\mu$ is the shear modulus, $\textbf{I}$ is the Identity matrix, $p$
# is the perturbation pressure, and $\boldsymbol{\sigma}_{L1} $ is the incremental lagrangian stress tensor. Note $p$
# is a perturbation pressure as we have already removed the hydostatic background state earlier. An overhead dot notes the
# time derivative i.e the viscous strain rate is proportional to stress, while the elastic strain  is proportional to stress.
# The total strain is
# \begin{equation}
#     \boldsymbol{\epsilon} = \boldsymbol{\epsilon}^v + \boldsymbol{\epsilon}^e = \dfrac{1}{2} ( \nabla \textbf{u} + (\nabla \textbf{u})^T),
# \end{equation}
# where $\textbf{u}$ is the displacement vector.
# Taking the time derivative of the total strain and substituting this into the consitutive equations gives
# \begin{equation}
#     \boldsymbol{\sigma}_{L1}+ \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}}_{L1} = - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}}.
# \end{equation}
#

# Summary
# --------
# The linearised governing equations for an incompressible Maxwell body used by the GIA community, and adopted herein, are
# \begin{equation}
#     \nabla \cdot \boldsymbol{\sigma}_{L1} - \nabla (\rho_0 g u_r)   - \rho_1 \textbf{g}  = 0,
# \end{equation}
# \begin{equation}
#     \nabla \cdot \textbf{v} = 0,
# \end{equation}
# \begin{equation}
#     \rho_1 = - \textbf{u} \cdot \nabla \rho_0,
# \end{equation}
# \begin{equation}
#     \boldsymbol{\sigma}_{L1}+ \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}}_{L1} = - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}} .
# \end{equation}
# Note as stated above, this still neglects perturbations in the gravitational field and we will leave solving the
# associated Poisson equation to a later demo.
#

# Time discretisation
# -------------------
#
# One of the key differences with the mantle convection demos is that the constitutive equation now depends on time.
# G-ADOPT implements the method of Zhong et al. (2003) where deviatoric stress is accounted for via an
# 'incremental displacement', thus recasting the problem in terms of $\textbf{u}_{inc}^n = \textbf{u}^n - \textbf{u}^{n-1}$,
# where subscripts refer to time levels $t$ and $t - \Delta t$ respectively.
# The incremental strain $\Delta \boldsymbol{\epsilon}$ is
# \begin{equation}
#     \Delta \boldsymbol{\epsilon} =   \dfrac{1}{2} ( \nabla \textbf{u}_{inc} + (\nabla \textbf{u}_{inc})^T).
# \end{equation}
#
#
# The constitutive equation is discretised by integrating from $t - \Delta t$ to $t$ using the trapezoid rule
# \begin{equation}
#     \int_{t-\Delta t}^t \boldsymbol{\sigma}_{L1} + \dfrac{\eta}{ \mu} \overset{\cdot}{\boldsymbol{\sigma}_{L1}} \, dt = \int_{t-\Delta t}^t - \left(p + \dfrac{\eta}{ \mu}\overset{\cdot}{p}\right) \textbf{I} + 2 \eta \overset{\cdot}{\boldsymbol{\epsilon}} \, dt.
# \end{equation}
# The trapezoid rule is
# \begin{equation}
#     \int_{a}^b f(x) \approx \dfrac{1}{2}(b - a)(f(a) + f(b).
# \end{equation}
# Using this gives
# \begin{equation}
#     \dfrac{\Delta t}{2} (\boldsymbol{\sigma}_{L1}^n + \boldsymbol{\sigma}_{L1}^{n-1}) + \dfrac{\eta}{ \mu} (\boldsymbol{\sigma}_{L1}^n - \boldsymbol{\sigma}_{L1}^{n-1}) = -\dfrac{\Delta t}{2} (p^n + p^{n-1}) \textbf{I} - \dfrac{\eta}{ \mu}(p^n - p^{n-1})  \textbf{I} + 2 \eta (\boldsymbol{\epsilon}^n - \boldsymbol{\epsilon}^{n-1}).
# \end{equation}
# Using Maxwell time, $\alpha = \eta / \mu$, this simplifies to
# \begin{equation}
#     \boldsymbol{\sigma}_{L1}^n  = - p^n \textbf{I} + \dfrac{2 \eta}{\alpha + \Delta t / 2}  \Delta \boldsymbol{\epsilon}^n + \dfrac{\alpha - \Delta t / 2}{\alpha + \Delta t / 2}(\boldsymbol{\sigma}_{L1}^{n-1} + p^{n-1} \textbf{I}).
# \end{equation}
#
# This expression for the stress is similar to that relevant for mantle convection. We are solving for incremental
# displacement instead of velocity, but the only difference between these two functions is the timestep multiplication factor.
# We also have a modified viscosity based on the timestep and the Maxwell time. Finally, the stress history is included as the
# last term is the deviatoric stress from the previous timestep multiplied by a prefactor involving the timestep and the Maxwell time.
# Note that in the small dt limit (i.e. $dt$ << $\alpha$) the effective viscosity tends towards the shear modulus. i.e. we are solving the elastic equations.

# This example
# -------------
# We will simulate a viscoelastic loading and unloading problem based on a 2D version of the test case presented in Weerdesteijn et al. (2023).
#
# Let's get started! The first step is to import the `gadopt` module, which
# provides access to Firedrake and associated functionality.

from gadopt import *
from gadopt.utility import step_func

# Next we need to create a mesh of the mantle region we want to simulate. The Weerdesteijn test case is a 3D box 1500 km wide horizontally and
# 2891 km deep. To speed up things for this first demo, we consider a 2D domain, i.e. taking a vertical cross section through the 3D box.
#
# For starters let's use one of the default meshes provided by Firedrake, `RectangleMesh`. We have chosen 40 quadrilateral elements in the $x$
# direction and 40 quadrilateral elements in the $y$ direction. It is worth emphasising that the setup has coarse grid resolution so that the
# demo is quick to run! For real simulations we can use fully unstructured meshes to accurately resolve important features in the model, for
# instance near coastlines or sharp discontinuities in mantle properties.  We can print out the grid resolution using `log`, a utility provided by
# G-ADOPT. (N.b. `log` is equivalent to python's `print` function, except that it simplifies outputs when running simulations in parallel.)
#
# On the mesh, we also denote that our geometry is Cartesian, i.e. gravity points
# in the negative z-direction. This attribute is used by G-ADOPT specifically, not
# Firedrake. By contrast, a non-Cartesian geometry is assumed to have gravity
# pointing in the radially inward direction.
#

# Boundaries are automatically tagged by the built-in meshes supported by Firedrake. For the `RectangleMesh` being used here, tag 1 corresponds
# to the plane $x=0$; 2 to the $x=L$ plane; 3 to the $y=0$ plane; and 4 to the $y=D$ plane. The `get_boundary_ids` function will inspect the mesh
# to detect this and allow the boundaries to be referred to more intuitively (e.g. `boundary.left`, `boundary.top`, etc.).

# +
# Set up geometry:
L = 1500e3  # Length of the domain in m
D = 2891e3  # Depth of domain in m
nx = 40  # Number of horizontal cells
nz = 40  # Number of vertical cells

# Let's print out the grid resolution in km
log(f"Horizontal resolution {L/nx/1000} km")
log(f"Vertical resolution {D/nz/1000} km")

mesh = RectangleMesh(nx, nz, L, D, name="mesh", quadrilateral=True)
mesh.cartesian = True

boundary = get_boundary_ids(mesh)
# -
# We now need to choose finite element function spaces. `V` , `W`, `S` and `R` are symbolic
# variables representing function spaces. They also contain the
# function space's computational implementation, recording the
# association of degrees of freedom with the mesh and pointing to the
# finite element basis. We will choose Q2-Q1 for the mixed incremental displacement-pressure similar to our mantle convection demos.
# This is a Taylor-Hood element pair which has good properties for Stokes modelling. We also initialise a discontinuous tensor function
# space that wil store our previous values of the deviatoric stress, as the gradient of the continous incremental displacement field will
# be discontinuous.

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # (Incremental) Displacement function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
S = TensorFunctionSpace(mesh, "DQ", 2)  # (Discontinuous) Stress tensor function space (tensor)
R = FunctionSpace(mesh, "R", 0)  # Real function space (for constants)

# Function spaces can be combined in the natural way to create mixed
# function spaces, combining the incremental displacement and pressure spaces to form
# a function space for the mixed Stokes problem, `Z`.

Z = MixedFunctionSpace([V, W])  # Mixed function space.

# We also specify functions to hold our solutions: `z` in the mixed
# function space, noting that a symbolic representation of the two
# parts – incremental displacement and pressure – is obtained with `split`. For later
# visualisation, we rename the subfunctions of `z` to *Incremental Displacement* and *Pressure*.
#
# We also need to initialise two functions `displacement` and `stress_old` that are used when timestepping the constitutive equation.

# +
z = Function(Z)  # A field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p
# Next rename for output:
z.subfunctions[0].rename("Incremental Displacement")
z.subfunctions[1].rename("Pressure")

displacement = Function(V, name="displacement").assign(0)
stress_old = Function(S, name="stress_old").assign(0)
# -

# We can output function space information, for example the number of degrees
# of freedom (DOF).

# Output function space information:
log("Number of Incremental Displacement DOF:", V.dim())
log("Number of Pressure DOF:", W.dim())
log("Number of Velocity and Pressure DOF:", V.dim()+W.dim())

# Let's start initialising some parameters. First of all Firedrake has a helpful function to give a symbolic representation of the mesh coordinates.

X = SpatialCoordinate(mesh)

# Now we can set up the background profiles for the material properties.
# In this case the density, shear modulus and viscosity only vary in the vertical direction.
# We will approximate the series of layers using a smooth tanh function with a width of 20 km.
# The layer properties specified are from spada et al. (2011).
# N.b. that we have modified the viscosity of the Lithosphere viscosity from
# Spada et al. (2011) because we are using coarse grid resolution.


# +
radius_values = [6371e3, 6301e3, 5951e3, 5701e3]
density_values = [3037, 3438, 3871, 4978]
shear_modulus_values = [0.50605e11, 0.70363e11, 1.05490e11, 2.28340e11]
viscosity_values = [1e25, 1e21, 1e21, 2e21]


def initialise_background_field(field, background_values, vertical_tanh_width=20e3):
    profile = background_values[0]
    sharpness = 1 / vertical_tanh_width
    depth = X[1] - D
    for i in range(1, len(background_values)):
        centre = radius_values[i] - radius_values[0]
        mag = background_values[i] - background_values[i-1]
        profile += step_func(depth, centre, mag, increasing=False, sharpness=sharpness)

    field.interpolate(profile)


density = Function(W, name="density")
initialise_background_field(density, density_values)

shear_modulus = Function(W, name="shear modulus")
initialise_background_field(shear_modulus, shear_modulus_values)

viscosity = Function(W, name="viscosity")
initialise_background_field(viscosity, viscosity_values)
# -

# Next let's define the length of our time step. If we want to accurately resolve the elastic response we should choose a
# timestep lower than the Maxwell time, $\alpha = \eta / \mu$. The Maxwell time is the time taken for the viscous deformation
# to 'catch up' with the initial, instantaneous elastic deformation.
#
# Let's print out the Maxwell time for each layer

year_in_seconds = 8.64e4 * 365.25
for layer_visc, layer_mu in zip(viscosity_values, shear_modulus_values):
    log(f"Maxwell time: {float(layer_visc/layer_mu/year_in_seconds):.0f} years")

# As we can see the shortest Maxwell time is given by the lower mantle and is about 280 years, i.e. it will take about 280
# years for the viscous deformation in that layer to catch up any instantaneous elastic deformation. Conversely the top layer,
# our lithosphere, has a Maxwell time of 6 million years. Given that our simulations only run for 110000 years the viscous
# deformation over the course of the simulation will always be negligible compared with the elastic deformation. For now let's
# choose a timestep of 250 years and an output frequency of 2000 years.

# +
# Timestepping parameters
Tstart = 0
time = Function(R).assign(Tstart * year_in_seconds)

dt_years = 250
dt = Constant(dt_years * year_in_seconds)
Tend_years = 110e3
Tend = Constant(Tend_years * year_in_seconds)
dt_out_years = 2e3
dt_out = Constant(dt_out_years * year_in_seconds)

max_timesteps = round((Tend - Tstart * year_in_seconds) / dt)
log("max timesteps: ", max_timesteps)

output_frequency = round(dt_out / dt)
log("output_frequency:", output_frequency)
log(f"dt: {float(dt / year_in_seconds)} years")
log(f"Simulation start time: {Tstart} years")
# -

# Next let's setup our ice load. Following the long test from Weeredesteijn et al 2023,
# during the first 90 thousand years of the simulation the ice sheet will grow to a thickness of 1 km.
# The ice thickness will rapidly shrink to ice free conditions in the next 10 thousand years. Finally,
# the simulation will run for a further 10 thousand years to allow the system to relax towards
# isostatic equilibrium. This is approximately the length of an interglacial-glacial cycle. The
# width of the ice sheet is 100 km and we have used a tanh function again to smooth out the
# transition from ice to ice-free regions.
#
# As the loading and unloading cycle only varies linearly in time, let's write the ice load as a symbolic expression.

# Initialise ice loading
rho_ice = 931
g = 9.8125
Hice = 1000
t1_load = 90e3 * year_in_seconds
t2_load = 100e3 * year_in_seconds
ramp_after_t1 = conditional(
    time < t2_load, 1 - (time - t1_load) / (t2_load - t1_load), 0
)
ramp = conditional(time < t1_load, time / t1_load, ramp_after_t1)
# Disc ice load but with a smooth transition given by a tanh profile
disc_radius = 100e3
disc_dx = 5e3
k_disc = 2*pi/(8*disc_dx)  # wavenumber for disk 2pi / lambda
r = X[0]
disc = 0.5*(1-tanh(k_disc * (r - disc_radius)))
ice_load = ramp * rho_ice * g * Hice * disc

# We can now define the boundary conditions to be used in this simulation.  Let's set the bottom and
# side boundaries to be free slip with no normal flow $\textbf{u} \cdot \textbf{n} =0$. By passing
# the string `ux` and `uy`, G-ADOPT knows to specify these as Strong Dirichlet boundary conditions.
#
# For the top surface we need to specify a normal stress, i.e. the weight of the ice load, as well as
# indicating this is a free surface.
#
# The `delta_rho_fs` option accounts for the density contrast across the free surface whether there
# is ice or air above a particular region of the mantle.

# +
# Setup boundary conditions
exterior_density = conditional(time < t2_load, rho_ice*disc, 0)
stokes_bcs = {
    boundary.bottom: {'uy': 0},
    boundary.top: {'normal_stress': ice_load, 'free_surface': {'delta_rho_fs': density - exterior_density}},
    boundary.left: {'ux': 0},
    boundary.right: {'ux': 0},
}

gd = GeodynamicalDiagnostics(z, density, boundary.bottom, boundary.top)
# -


# We also need to specify a G-ADOPT approximation which sets up the various parameters and fields
# needed for the viscoelastic loading problem.


approximation = SmallDisplacementViscoelasticApproximation(density, shear_modulus, viscosity, g=g)

# We finally come to solving the variational problem, with solver
# objects for the Stokes system created. We pass in the solution fields `z` and various fields
# needed for the solve along with the approximation, timestep and boundary conditions.
#

stokes_solver = ViscoelasticStokesSolver(z, stress_old, displacement, approximation,
                                         dt, bcs=stokes_bcs)

# We next set up our output, in VTK format. This format can be read by programs like pyvista and Paraview.

# +
# Create output file
output_file = VTKFile("output.pvd")
output_file.write(*z.subfunctions, displacement)

plog = ParameterLog("params.log", mesh)
plog.log_str(
    "timestep time dt u_rms u_rms_surf ux_max disp_min disp_max"
)

checkpoint_filename = "viscoelastic_loading-chk.h5"
# -

# Now let's run the simulation! We are going to control the ice thickness using the `ramp` parameter.
# At each step we call `solve` to calculate the incremental displacement and pressure fields. This
# will update the displacement at the surface and stress values accounting for the time dependent
# Maxwell consitutive equation.

for timestep in range(max_timesteps):
    stokes_solver.solve()

    time.assign(time+dt)

    if timestep % output_frequency == 0:
        log("timestep", timestep)

        output_file.write(*z.subfunctions, displacement)

        with CheckpointFile(checkpoint_filename, "w") as checkpoint:
            checkpoint.save_function(z, name="Stokes")
            checkpoint.save_function(displacement, name="Displacement")
            checkpoint.save_function(stress_old, name="Deviatoric stress")

    # Log diagnostics:
    plog.log_str(
        f"{timestep} {float(time)} {float(dt)} "
        f"{gd.u_rms()} {gd.u_rms_top()} {gd.ux_max(boundary.top)} "
        f"{displacement.dat.data[:, 1].min()} {displacement.dat.data[:, 1].max()}"
    )

# Let's use the python package *PyVista* to plot the magnitude of the displacement field through time.
# We will use the calculated displacement to artifically scale the mesh. We have exaggerated the stretching
# by a factor of 1500, **BUT...** it is important to remember this is just for ease of visualisation -
# the mesh is not moving in reality!

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import pyvista as pv
#
# # Read the PVD file
# reader = pv.get_reader("output.pvd")
# data = reader.read()[0]  # MultiBlock mesh with only 1 block
#
# # Create a plotter object
# plotter = pv.Plotter(shape=(1, 1), border=False, notebook=True, off_screen=False)
#
# # Open a gif
# plotter.open_gif("displacement_warp.gif")
#
# # Make a colour map
# boring_cmap = plt.get_cmap("viridis", 25)
#
# for i in range(len(reader.time_values)):
#     reader.set_active_time_point(i)
#     data = reader.read()[0]
#
#     # Artificially warp the output data in the vertical direction by the free surface height
#     # Note the mesh is not really moving!
#     warped = data.warp_by_vector(vectors="displacement", factor=1500)
#     arrows = data.glyph(orient="Incremental Displacement", scale="Incremental Displacement", factor=400000, tolerance=0.05)
#     plotter.add_mesh(arrows, color="white", lighting=False)
#
#     # Add the warped displacement field to the frame
#     plotter.add_mesh(
#         warped,
#         scalars="displacement",
#         component=None,
#         lighting=False,
#         show_edges=False,
#         clim=[0, 70],
#         cmap=boring_cmap,
#         scalar_bar_args={
#             "title": 'Displacement (m)',
#             "position_x": 0.8,
#             "position_y": 0.2,
#             "vertical": True,
#             "title_font_size": 20,
#             "label_font_size": 16,
#             "fmt": "%.0f",
#             "font_family": "arial",
#         }
#     )
#
#     # Fix camera in default position otherwise mesh appears to jump around!
#     plotter.camera_position = [(750000.0, 1445500.0, 6291991.008627122),
#                         (750000.0, 1445500.0, 0.0),
#                         (0.0, 1.0, 0.0)]
#     plotter.add_text(f"Time: {i*2000:6} years", name='time-label')
#     plotter.write_frame()
#
#     if i == len(reader.time_values)-1:
#         # Write end frame multiple times to give a pause before gif starts again!
#         for j in range(20):
#             plotter.write_frame()
#
#     plotter.clear()
#
# # Closes and finalizes movie
# plotter.close()
# -
# Looking at the animation, we can see that as the weight of the ice load builds up the mantle deforms,
# pushing up material away from the ice load. If we kept the ice load fixed this forebulge will
# eventually grow enough that it balances the weight of the ice, i.e the mantle is in isostatic
# equilbrium and the deformation due to the ice load stops. At 100 thousand years when the ice is removed
# the topographic highs associated with forebulges are now out of equilibrium so the flow of material
# in the mantle reverses back towards the previously glaciated region.

# ![SegmentLocal](displacement_warp.gif "segment")

# References
# ----------
# Cathles L.M. (1975). *Viscosity of the Earth's Mantle*, Princeton University Press.
#
# Dahlen F. A. and Tromp J. (1998). *Theoretical Global Seismology*, Princeton University Press.
#
# Ranalli, G. (1995). Rheology of the Earth. Springer Science & Business Media.
#
# Weerdesteijn, M. F., Naliboff, J. B., Conrad, C. P., Reusen, J. M., Steffen, R., Heister, T., &
# Zhang, J. (2023). *Modeling viscoelastic solid earth deformation due to ice age and contemporary
# glacial mass changes in ASPECT*. Geochemistry, Geophysics, Geosystems.
#
# Wu P., Peltier W. R. (1982). *Viscous gravitational relaxation*, Geophysical Journal International.
#
# Zhong, S., Paulson, A., & Wahr, J. (2003). Three-dimensional finite-element modelling of Earth’s
# viscoelastic deformation: effects of lateral variations in lithospheric thickness. Geophysical
# Journal International.
