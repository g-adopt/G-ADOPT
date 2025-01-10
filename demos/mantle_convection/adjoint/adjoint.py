# Adjoint inverse reconstruction
# ==============================
#
# Introduction
# ------------
# In this tutorial, we will demonstrate how to perform an inversion to recover the initial temperature field of an
# idealised mantle convection simulation using G-ADOPT. This tutorial is published as the first synthetic experiment in
# *Ghelichkhan et al. (2024)*. The full inversion showcased in the publication involves a total number of 80 timesteps.
# For the tutorial here we start with only 5 timesteps to go through the basics.
#
# The tutorial involves a *twin experiment*, where we assess the performance of the inversion scheme by inverting the
# initial state of a synthetic reference simulation, known as the "*Reference Twin*". To create this reference twin, we
# run a forward mantle convection simulation and record all relevant fields (velocity and temperature) at each time step.
#
# We have pre-run this simulation by running [the forward case](../adjoint_forward), and stored model output as a
# checkpoint file on our servers.  These fields serve as benchmarks for evaluating our inverse problem's performance. To
# download the reference benchmark checkpoint file if it doesn't already exist, execute the following command:

# + tags=["active-ipynb"]
# ![ ! -f adjoint-demo-checkpoint-state.h5 ] && wget https://data.gadopt.org/demos/adjoint-demo-checkpoint-state.h5
# -

# In this file, fields from the reference simulation are stored under the names "Temperature" and "Velocity".
# After importing g-adopt and the associated inverse module (gadopt.inverse - discussed further below), we can
# retrieve timestepping information from the pre-computed forward run as follows

# +
from gadopt import *
from gadopt.inverse import *

# Open the checkpoint file and subsequently load the mesh:
checkpoint_filename = "adjoint-demo-checkpoint-state.h5"
checkpoint_file = CheckpointFile(checkpoint_filename, mode="r")
mesh = checkpoint_file.load_mesh("firedrake_default_extruded")
mesh.cartesian = True

# Specify boundary markers, noting that for extruded meshes the upper and lower boundaries are tagged as
# "top" and "bottom" respectively.
boundary = get_boundary_ids(mesh)

# Retrieve the timestepping information for the Velocity and Temperature functions from checkpoint file:
temperature_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Temperature")
velocity_timestepping_info = checkpoint_file.get_timestepping_history(mesh, "Velocity")
# -

# We can check the information for each:

# + tags=["active-ipynb"]
# print("Timestepping info for Temperature", temperature_timestepping_info)
# print("Timestepping info for Velocity", velocity_timestepping_info)
# -

# The timestepping information reveals that there are 80 time-steps (from 0 to 79) in the reference simulation,
# with the temperature field stored only at the initial (index=0) and final (index=79) timesteps, while the
# velocity field is stored at all timesteps. We can visualise the benchmark fields using Firedrake's built-in VTK
# functionality. For example, initial and final temperature fields can be loaded:

# Load the final state, analagous to the present-day "observed" state:
Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][-1]))
Tobs.rename("Observed Temperature")
# Load the reference initial state - i.e. the state that we wish to recover:
Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=int(temperature_timestepping_info["index"][0]))
Tic_ref.rename("Reference Initial Temperature")
checkpoint_file.close()

# These fields can be visualised using standard VTK software, such as Paraview or pyvista.

# + tags=["active-ipynb"]
# import pyvista as pv
# VTKFile("./visualisation_vtk.pvd").write(Tobs, Tic_ref)
# dataset = pv.read('./visualisation_vtk.pvd')
# # Create a plotter object
# plotter = pv.Plotter()
# # Add the dataset to the plotter
# plotter.add_mesh(dataset, scalars='Observed Temperature', cmap='coolwarm')
# # Adjust the camera position
# plotter.camera_position = [(0.5, 0.5, 2.5), (0.5, 0.5, 0), (0, 1, 0)]
# # Show the plot
# plotter.show(jupyter_backend="static")
# -

# The Inverse Code
# ----------------
#
# The novelty of using the overloading approach provided by pyadjoint is that it requires
# minimal changes to our script to enable the inverse capabalities of G-ADOPT.
# To turn on the adjoint, one simply imports the inverse module (already done above) to
# enable all taping functionality from pyadjoint.
#
# Doing so will turn Firedrake's objects to overloaded types, in a way
# that any UFL operation will be annotated and added to the tape, unless
# otherwise specified.
#
# We first ensure that the tape is cleared of any previous operations, using the following code:

tape = get_working_tape()
tape.clear_tape()

# + tags=["active-ipynb"]
# # To verify the tape is empty, we can print all blocks:
# print(tape.get_blocks())
# -

# From here on, all user operations are specified with minimal differences relative to
# to our forward code. Under the hood, however, the tape will be populated
# by *blocks* that record their dependencies. Knowing the mesh was loaded above, we continue
# in a manner that is consistent with our most basic forward modelling tutorials.

# +
# Set up function spaces:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space

# Specify test functions and functions to hold solutions:
z = Function(Z)  # A field over the mixed function space Z
u, p = split(z)  # Returns symbolic UFL expression for u and p
z.subfunctions[0].rename("Velocity")
z.subfunctions[1].rename("Pressure")
T = Function(Q, name="Temperature")

# Specify important constants for the problem, alongside the approximation:
Ra = Constant(1e6)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# Define time-stepping parameters:
delta_t = Constant(4e-6)  # Constant time step
timesteps = int(temperature_timestepping_info["index"][-1]) + 1  # number of timesteps from forward

# Nullspaces for the problem are next defined:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Followed by boundary conditions, noting that all boundaries are free slip, whilst the domain is
# heated from below (T = 1) and cooled from above (T = 0).
stokes_bcs = {
    boundary.bottom: {"uy": 0},
    boundary.top: {"uy": 0},
    boundary.left: {"ux": 0},
    boundary.right: {"ux": 0},
}
temp_bcs = {
    boundary.bottom: {"T": 1.0},
    boundary.top: {"T": 0.0},
}

# Setup Energy and Stokes solver
energy_solver = EnergySolver(T, u, approximation, delta_t, ImplicitMidpoint, bcs=temp_bcs)
stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace, constant_jacobian=True)
# -

# Specify Problem Length
# ------------------------
#
# For the purpose of this demo, we only invert for a total of 10 time-steps. This makes it
# tractable to run this within a tutorial session.
#
# To run for the simulation's full duration, change the initial_timestep to `0` below, rather than
# `timesteps - 10`.

initial_timestep = timesteps - 10

# Define the Control Space
# ------------------------
#
# In this section, we define the control space, which can be restricted to reduce the risk of encountering an
# undetermined problem. Here, we select the Q1 function space for the initial condition $T_{ic}$. We also provide an
# initial guess for the control value, which in this synthetic test is the temperature field of the reference
# simulation at the final time-step (`timesteps - 1`). In other words, our guess for the initial temperature
# is the final model state.

# +
# Define control function space:
Q1 = FunctionSpace(mesh, "CG", 1)

# Create a function for the unknown initial temperature condition, which we will be inverting for. Our initial
# guess is set to the 1-D average of the forward model. We first load that, at the relevant timestep.
# Note that this layer average will later be used for the smoothing term in our objective functional.
with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
    Taverage = checkpoint_file.load_function(mesh, "Average_Temperature", idx=initial_timestep)
Tic = Function(Q1, name="Initial_Condition_Temperature").assign(Taverage)

# Given that Tic will be updated during the optimisation, we also create a function to store our initial guess,
# which we will later use for smoothing. Note that since smoothing is executed in the control space, we must
# specify boundary conditions on this term in that same Q1 space.
T0_bcs = [DirichletBC(Q1, 0., boundary.top), DirichletBC(Q1, 1., boundary.bottom)]
T0 = Function(Q1, name="Initial_Guess_Temperature").project(Tic, bcs=T0_bcs)

# We next make pyadjoint aware of our control problem:
control = Control(Tic)

# Take our initial guess and project to T, simultaneously applying boundary conditions in the Q2 space:
T.project(Tic, bcs=energy_solver.strong_bcs)

# We continue by integrating the solutions at each time-step.
# Notice that we cumulatively compute the misfit term with respect to the
# surface velocity observable.

# +
u_misfit = 0.0

# Next populate the tape by running the forward simulation.
for time_idx in range(initial_timestep, timesteps):
    stokes_solver.solve()
    energy_solver.solve()
    # Update the accumulated surface velocity misfit using the observed value.
    with CheckpointFile(checkpoint_filename, mode="r") as checkpoint_file:
        uobs = checkpoint_file.load_function(mesh, name="Velocity", idx=time_idx)
    u_misfit += assemble(dot(u - uobs, u - uobs) * ds_t)
# -

# Define the Objective Functional
# -------------------------------
#
# Now that all calculations are in place, we must define *the objective functional*.
# The objective functional is our way of expressing our goal for this optimisation.
# It is composed of several terms, each representing a different aspect of the model's
# performance and regularisation.
#
# Regularisation involves imposing constraints on solutions to prevent overfitting, ensuring that the model
# generalises well to new data. In this context, we use the one-dimensional (1-D) temperature profile derived from
# the reference simulation as our regularisation constraint. This profile, referred to below as `Taverage`, helps
# stabilise the inversion process by providing a benchmark that guides the solution towards physically plausible states.
#
# We use `Taverage` as a part of the damping and smoothing terms in our regularisation.
# Consequently, the complete objective functional is defined mathematically as follows:
#
# Reiterating that:
# - $T_{ic}$ is the initial temperature condition.
# - $T_{\text{average}}$ is the average temperature profile representing mantle's geotherm.
# - $T_{F}$ is the the temperature field at final time-step.
# - $T_{\text{obs}}$ is the observed temperature field at the final time-step.
# - $u_{\text{obs}}$ is the observed velocity field at *each time-step*.
# - $\alpha_u$, $\alpha_d$, $\alpha_s$ are the three different
#   weighting terms for the velocity, damping and smoothing terms.
#
# We define the objective functional as
# $$ \text{Objective Functional}= \int_{\Omega}(T - T_{\text{obs}}) ^ 2 \, dx \\
#                  +\alpha_u\, \frac{D_{T_{obs}}}{N\times D_{u_{obs}}}\sum_{i}\int_{\partial \Omega_{\text{top}}}(u - u_{\text{obs}}) \cdot(u - u_{\text{obs}}) \, ds \\
#                  +\alpha_s\, \frac{D_{T_{obs}}}{D_{\text{smoothing}}}\int_{\Omega} \nabla(T_{ic} - T_{\text{average}}) \cdot \nabla(T_{ic} - T_{\text{average}}) \, dx \\
#                  +\alpha_d\, \frac{D_{T_{obs}}}{D_{\text{damping}}}\int_{\Omega}(T_{ic} - T_{\text{average}}) ^ 2 \, dx $$

# With the three *normlisation terms* of:
# + $D_{\text{damping}} = \int_{\Omega} T_{\text{average}}^2 \, dx$,
# + $D_{\text{smoothing}} = \int_{\Omega} \nabla T_{\text{obs}} \cdot \nabla T_{\text{obs}} \, dx$,
# + $D_{T_{obs}} = \int_{\Omega} T_{\text{obs}} ^ 2 \, dx$, and
# + $D_{\text{damping}} = \int_{\partial \Omega_{\text{top}}} u_{\text{obs}} \cdot u_{\text{obs}} \, ds$
#
# which we specify through the `objective` below:

# +
# Define component terms of overall objective functional and their normalisation terms:
damping = assemble((T0 - Taverage) ** 2 * dx)
norm_damping = assemble(Taverage**2 * dx)
smoothing = assemble(dot(grad(T0 - Taverage), grad(T0 - Taverage)) * dx)
norm_smoothing = assemble(dot(grad(Tobs), grad(Tobs)) * dx)
norm_obs = assemble(Tobs**2 * dx)
norm_u_surface = assemble(dot(uobs, uobs) * ds_t)

# Define temperature misfit between final state solution and observation:
t_misfit = assemble((T - Tobs) ** 2 * dx)

# Weighting terms
alpha_u = 1e-1
alpha_d = 1e-3
alpha_s = 1e-3

# Define overall objective functional:
objective = (
    t_misfit +
    alpha_u * (norm_obs * u_misfit / timesteps / norm_u_surface) +
    alpha_d * (norm_obs * damping / norm_damping) +
    alpha_s * (norm_obs * smoothing / norm_smoothing)
)
# -

# Define the Reduced Functional
# -----------------------------
#
# In optimisation terminology, a reduced functional is a functional that takes a given value for the control and outputs
# the value of the objective functional defined for it. It does this without explicitly depending on all intermediary
# state variables, hence the name "reduced".
#
# To define the reduced functional, we provide the class with an objective (which is an overloaded UFL object) and the control.

reduced_functional = ReducedFunctional(objective, control)

# At this point, we have completed annotating the tape with the necessary information from running the forward simulation.
# To prevent further annotations during subsequent operations, we stop the annotation process. This ensures that no additional
# solves are unnecessarily recorded, keeping the tape focused only on the essential steps.

pause_annotation()

# We can print the contents of the tape at this stage to verify that it is not empty.

# + tags=["active-ipynb"]
# print(tape.get_blocks())
# -

# Verification of Gradients: Taylor Remainder Convergence Test
# ------------------------------------------------------------
#
# A fundamental tool for verifying gradients is the Taylor remainder convergence test. This test helps ensure that
# the gradients computed by our optimisation algorithm are accurate. For the reduced functional, $J(T_{ic})$, and its derivative,
# $\frac{\mathrm{d} J}{\mathrm{d} T_{ic}}$, the Taylor remainder convergence test can be expressed as:
#
# $$ \left| J(T_{ic} + h \,\delta T_{ic}) - J(T_{ic}) - h\,\frac{\mathrm{d} J}{\mathrm{d} T_{ic}} \cdot \delta T_{ic} \right| \longrightarrow 0 \text{ at } O(h^2). $$
#
# The expression on the left-hand side is termed the second-order Taylor remainder. This term's convergence rate of $O(h^2)$ is a robust indicator for
# verifying the computational implementation of the gradient calculation. Essentially, if you halve the value of $h$, the magnitude
# of the second-order Taylor remainder should decrease by a factor of 4.
#
# We employ these so-called *Taylor tests* to confirm the accuracy of the determined gradients. The theoretical convergence rate is
# $O(2.0)$, and achieving this rate indicates that the gradient information is accurate down to floating-point precision.
#
# ### Performing Taylor Tests
#
# In our implementation, we perform a second-order Taylor remainder test for each term of the objective functional. The test involves
# computing the functional and the associated gradient when randomly perturbing the initial temperature field, $T_{ic}$, and subsequently
# halving the perturbations at each level.
#
# Here is how you can perform a Taylor test in the code:

# + tags=["active-ipynb"]
# # Define the perturbation in the initial temperature field
# import numpy as np
# Delta_temp = Function(Tic.function_space(), name="Delta_Temperature")
# Delta_temp.dat.data[:] = np.random.random(Delta_temp.dat.data.shape)
#
# # Perform the Taylor test to verify the gradients
# minconv = taylor_test(reduced_functional, Tic, Delta_temp)
# -

# The `taylor_test` function computes the Taylor remainder and verifies that the convergence rate is close to the theoretical value of $O(2.0)$. This ensures
# that our gradients are accurate and reliable for optimisation.

# Running the inversion
# ---------------------
# In the final section of this tutorial, we run the optimisation method. First, we define lower and upper bounds for the optimisation problem to guide
# the optimisation method towards a more constrained solution.
#
# For this simple problem, we perform a bounded nonlinear optimisation where the temperature is only permitted to lie in the range [0, 1]. This means that the
# optimisation problem should not search for solutions beyond these values.

# +
# Define lower and upper bounds for the temperature
T_lb = Function(Tic.function_space(), name="Lower Bound Temperature")
T_ub = Function(Tic.function_space(), name="Upper Bound Temperature")

# Assign the bounds
T_lb.assign(0.0)
T_ub.assign(1.0)

# Define the minimisation problem, with the goal to minimise the reduced functional
# Note: in some scenarios, the goal might be to maximise (rather than minimise) the functional.
minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))
# -

# Using the Lin-Moré optimiser
# ----------------------------
#
# In this tutorial, we employ the trust region method of Lin and Moré (1999) implemented in ROL (Rapid Optimization Library).
# Lin-Moré is a truncated Newton method, which involves the repeated application of an iterative algorithm to approximately
# solve Newton’s equations (Dembo and Steihaug, 1983).
#
# Lin-Moré effectively handles provided bound constraints by ensuring that variables remain within their specified bounds.
# During each iteration, variables are classified into "active" and "inactive" sets. Variables at their bounds that do not
# allow descent are considered active and are fixed during the iteration. The remaining variables, which can change without
# violating the bounds, are inactive. These properties make the algorithm robust and efficient for solving bound-constrained
# optimisation problems.
#
# For our solution of the optimisation problem we use the pre-defined paramters set in gadopt by using `minimsation_parameters`.
# Here, we set the number of iterations to only 5, as opposed to the default 100. We also adjust the step-length for this problem,
# by setting it to a lower value than our default.

minimisation_parameters["Status Test"]["Iteration Limit"] = 5
minimisation_parameters["Step"]["Trust Region"]["Initial Radius"] = 1e-2

# A notable feature of this optimisation approach in ROL is its checkpointing capability. For every iteration,
# all information necessary to restart the optimisation from that iteration is saved in the specified `checkpoint_dir`.

# Define the LinMore Optimiser class with checkpointing capability:
optimiser = LinMoreOptimiser(
    minimisation_problem,
    minimisation_parameters,
    checkpoint_dir="optimisation_checkpoint",
)

# For sake of book-keeping the simulation, we have also implemented a user-defined way of
# recording information that might be used to check the optimisation performance. This
# callback function will be executed at the end of each iteration. Here, we write out
# the control field, i.e., the reconstructed intial temperature field, at the end of
# each iteration. To access the last value of *an overloaded object* we should access the
# `.block_variable.checkpoint` method as below.
#
# For the sake of this demo, we also record the values of the reduced
# functional directly in order to produce a plot of the convergence.

# +
solutions_vtk = VTKFile("solutions.pvd")
solution_container = Function(Tic.function_space(), name="Solutions")
functional_values = []


def callback():
    solution_container.assign(Tic.block_variable.checkpoint)
    solutions_vtk.write(solution_container)
    final_temperature_misfit = assemble(
        (T.block_variable.checkpoint - Tobs) ** 2 * dx
    )
    log(f"Terminal Temperature Misfit: {final_temperature_misfit}")


def record_value(value, *args):
    functional_values.append(value)


optimiser.add_callback(callback)
reduced_functional.eval_cb_post = record_value

# If it existed, we could restore the optimisation from last checkpoint:
# optimiser.restore()

# Run the optimisation
optimiser.run()

# Write the functional values to a file
with open("functional.txt", "w") as f:
    f.write("\n".join(str(x) for x in functional_values))
# -

# At this point a total number of 5 iterations are performed. For the example
# case here with 10 timesteps this should result an adequete reduction
# in the objective functional. Now we can look at the solution
# visually. For the actual simulation with 80 time-steps, this solution
# could be compared to `Tic_ref` as the "true solution".

# + tags=["active-ipynb"]
# import pyvista as pv
# VTKFile("./solution.pvd").write(optimiser.rol_solver.rolvector.dat[0])
# dataset = pv.read('./solution.pvd')
# # Create a plotter object
# plotter = pv.Plotter()
# # Add the dataset to the plotter
# plotter.add_mesh(dataset, scalars=dataset[0].array_names[0], cmap='coolwarm')
# plotter.add_text("Solution after 5 iterations", font_size=10)
# # Adjust the camera position
# plotter.camera_position = [(0.5, 0.5, 2.5), (0.5, 0.5, 0), (0, 1, 0)]
# # Show the plot
# plotter.show(jupyter_backend="static")

# + tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# plt.plot(functional_values)
# plt.xlabel("Optimisation iteration")
# plt.ylabel("Reduced functional")
# plt.title("Optimisation convergence")
