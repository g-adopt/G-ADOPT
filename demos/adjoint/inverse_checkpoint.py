from gadopt import *
from gadopt.inverse import *

ds_t = ds_t(degree=6)
dx = dx(degree=6)


def main():
    inverse(alpha_u=1e-1, alpha_d=1e-2, alpha_s=1e-1)


def inverse(alpha_u, alpha_d, alpha_s):
    """
    Use adjoint-based optimisation to solve for the initial condition of the rectangular
    problem.

    Parameters:
        alpha_u: The coefficient of the velocity misfit term
        alpha_d: The coefficient of the initial condition damping term
        alpha_s: The coefficient of the smoothing term
    """

    # Clear the tape of any previous operations to ensure
    # the adjoint reflects the forward problem we solve here
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    enable_disk_checkpointing()

    bottom_id, top_id = "bottom", "top"
    left_id, right_id = 1, 2

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # control space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    z = Function(Z)  # A field over the mixed function space Z
    u, p = z.subfunctions
    u.rename("Velocity")
    p.rename("Pressure")
    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)  # Constant time step
    max_timesteps = 80
    init_timestep = 0

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")
    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        Tic.project(f.load_function(mesh, "Temperature", idx=max_timesteps - 1))

    # Temperature function in Q2, where we solve the equations
    T = Function(Q, name="Temperature")

    # Free-slip velocity boundary condition on all sides
    stokes_bcs = {
        bottom_id: {"uy": 0},
        top_id: {"uy": 0},
        left_id: {"ux": 0},
        right_id: {"ux": 0},
    }
    temp_bcs = {
        top_id: {"T": 0.0},
        bottom_id: {"T": 1.0},
    }

    energy_solver = EnergySolver(
        T,
        u,
        approximation,
        delta_t,
        ImplicitMidpoint,
        bcs=temp_bcs,
    )

    stokes_solver = StokesSolver(
        z,
        T,
        approximation,
        bcs=stokes_bcs,
        nullspace=Z_nullspace,
        transpose_nullspace=Z_nullspace,
    )

    # Control variable for optimisation
    control = Control(Tic)

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")
    u_misfit = 0

    # We need to project the initial condition from Q1 to Q2,
    # and impose the boundary conditions at the same time
    T.project(Tic, bcs=energy_solver.strong_bcs)

    # Populate the tape by running the forward simulation
    for timestep in range(init_timestep, max_timesteps):
        stokes_solver.solve()
        energy_solver.solve()

        # Update the accumulated surface velocity misfit using the observed value
        u_obs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep
        )
        u_misfit += assemble(dot(u - u_obs, u - u_obs) * ds_t)

    # Load the observed final state
    T_obs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    T_obs.rename("Observed Temperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    Tic_ref = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_ref.rename("Reference Initial Temperature")

    # Load the average temperature profile
    T_average = checkpoint_file.load_function(mesh, "Average Temperature", idx=0)

    checkpoint_file.close()

    # Define the component terms of the overall objective functional
    damping = assemble((Tic - T_average) ** 2 * dx)
    norm_damping = assemble(T_average ** 2 * dx)
    smoothing = assemble(dot(grad(Tic - T_average), grad(Tic - T_average)) * dx)
    norm_smoothing = assemble(dot(grad(T_obs), grad(T_obs)) * dx)
    norm_obs = assemble(T_obs ** 2 * dx)
    norm_u_surface = assemble(dot(u_obs, u_obs) * ds_t)

    # Temperature misfit between solution and observation
    t_misfit = assemble((T - T_obs) ** 2 * dx)

    objective = (
        t_misfit +
        alpha_u * (norm_obs * u_misfit / max_timesteps / norm_u_surface) +
        alpha_d * (norm_obs * damping / norm_damping) +
        alpha_s * (norm_obs * smoothing / norm_smoothing)
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    def callback():
        initial_misfit = assemble((Tic.block_variable.checkpoint.restore() - Tic_ref) ** 2 * dx)
        final_misfit = assemble((T.block_variable.checkpoint.restore() - T_obs) ** 2 * dx)

        log(f"Initial misfit; {initial_misfit}; final misfit: {final_misfit}")

    # Perform a bounded nonlinear optimisation where temperature
    # is only permitted to lie in the range [0, 1]
    T_lb = Function(Q1, name="Lower bound temperature")
    T_ub = Function(Q1, name="Upper bound temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    minimisation_problem = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    optimiser = LinMoreOptimiser(
        minimisation_problem,
        minimisation_parameters,
        checkpoint_dir="optimisation_checkpoint"
    )
    optimiser.add_callback(callback)
    optimiser.run()

    # If we're performing mulitple successive optimisations, we want
    # to ensure the annotations are switched back on for the next code
    # to use them
    continue_annotation()


if __name__ == "__main__":
    main()
