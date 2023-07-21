from gadopt import *
from firedrake_adjoint import *
from os import mkdir
from os.path import isdir


def main():
    inverse(alpha_u=1.0, alpha_d=1.0, alpha_s=1.0)


def inverse(alpha_u, alpha_d, alpha_s):

    # Make sure we start from a clean tape
    tape = get_working_tape()
    tape.clear_tape()

    with CheckpointFile("mesh.h5", "r") as f:
        mesh = f.load_mesh("firedrake_default_extruded")

    # Enable checkpointing
    enable_disk_checkpointing()

    bottom_id, top_id = "bottom", "top"
    left_id, right_id = 1, 2

    # Set up function spaces for the Q2Q1 pair
    V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
    W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
    Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
    Q1 = FunctionSpace(mesh, "CG", 1)  # control space (scalar, P1)
    Z = MixedFunctionSpace([V, W])

    q = TestFunction(Q)
    q1 = TestFunction(Q1)

    z = Function(Z)  # A field over the mixed function space Z
    u, p = split(z)  # Symbolic UFL expressions for u and p

    # spliting z space to acces velocity and pressure
    u, p = z.split()
    u.rename("Velocity")
    p.rename("Pressure")

    # Without a restart to continue from, our initial guess is the final state of the forward run
    # We need to project the state from Q2 into Q1
    Tic = Function(Q1, name="Initial Temperature")

    # Temperature function
    T = Function(Q, name="Temperature")

    Ra = Constant(1e6)
    approximation = BoussinesqApproximation(Ra)

    delta_t = Constant(4e-6)  # Constant time step
    max_timesteps = 80
    init_timestep = 0

    Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

    # the initial guess for the control
    with CheckpointFile("Checkpoint_State.h5", "r") as f:
        Tic.project(f.load_function(mesh, "Temperature", idx=max_timesteps - 1))

    # Imposed velocity boundary condition on top, free-slip on other sides
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
    temp_bcs_q1 = [
        DirichletBC(Q1, 0.0, top_id),
        DirichletBC(Q1, 1.0, bottom_id),
    ]

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

    # Define a simple problem to apply the imposed boundary condition to the IC
    T_ = Function(Tic.function_space())
    bc_problem = LinearVariationalProblem(
        q1 * TrialFunction(Tic.function_space()) * dx,
        q1 * T_ * dx,
        Tic,
        bcs=temp_bcs_q1,
    )
    bc_solver = LinearVariationalSolver(bc_problem)

    # Project the initial condition from Q1 to Q
    ic_projection_problem = LinearVariationalProblem(
        q * TrialFunction(Q) * dx,
        q * Tic * dx,
        T,
        bcs=energy_solver.strong_bcs,
    )
    ic_projection_solver = LinearVariationalSolver(ic_projection_problem)

    # Control variable for optimisation
    control = Control(Tic)

    # Apply the boundary condition to the control
    # and obtain the initial condition
    T_.assign(Tic)
    bc_solver.solve()
    ic_projection_solver.solve()

    checkpoint_file = CheckpointFile("Checkpoint_State.h5", "r")

    u_misfit = 0.0

    # Populate the tape by running the forward simulation
    for timestep in range(init_timestep, max_timesteps):
        stokes_solver.solve()

        # load the reference velocity
        uobs = checkpoint_file.load_function(
            mesh,
            name="Velocity",
            idx=timestep)
        u_misfit += assemble(0.5 * (uobs - u)**2 * ds_t)

        energy_solver.solve()

    # Load the observed final state
    Tobs = checkpoint_file.load_function(mesh, "Temperature", idx=max_timesteps - 1)
    Tobs.rename("ObservedTemperature")

    # Load the reference initial state
    # Needed to measure performance of weightings
    Tic_obs = checkpoint_file.load_function(mesh, "Temperature", idx=0)
    Tic_obs.rename("ReferenceInitialTemperature")

    # Load the average temperature profile
    Taverage = checkpoint_file.load_function(mesh, "AverageTemperature", idx=0)
    Taverage.rename("AverageTemperature")

    checkpoint_file.close()

    # Normalisation for smoothing term
    norm_grad_Taverage = assemble(
        0.5*dot(grad(Taverage), grad(Taverage)) * dx)
    # Normalisation for damping term
    norm_Tavereage = assemble(
        0.5*(Taverage)**2 * dx)
    # normlaisation for temperature
    norm_final_state = assemble(
        0.5*(Tobs)**2 * dx)
    # normalisation for surface velocities
    norm_u_surface = assemble(
        0.5 * (uobs)**2 * ds_t)

    objective = (
        assemble(0.5 * (T - Tobs)**2 * dx) +
        alpha_u * norm_final_state * u_misfit / max_timesteps / norm_u_surface +
        alpha_s * norm_final_state * assemble(
            0.5 * dot(grad(Tic-Taverage), grad(Tic-Taverage)) * dx) / norm_grad_Taverage +
        alpha_d * norm_final_state * assemble(0.5 * (Tic - Taverage)**2 * dx) / norm_Tavereage
    )

    # All done with the forward run, stop annotating anything else to the tape
    pause_annotation()

    reduced_functional = ReducedFunctional(objective, control)

    class ROL_callback(object):
        def __init__(self):
            # keeping track of iteration number
            # self.iteration = my_restarter.iteration
            self.iteration = 0

            # making sure solution directory exists
            if mesh.comm.rank == 0 and not isdir('./solutions'):
                mkdir('./solutions')

            mesh.comm.barrier()

        def __call__(self):
            # first checkpoint the optimisation for rerun
            log("\tFinal misfit: {}".format(
                assemble(
                    (T.block_variable.checkpoint.restore() -
                     Tobs)**2 * dx)))

            log("\tInitial misfit : {}".format(
                assemble(
                    (Tic.block_variable.checkpoint.restore() -
                     Tic_obs)**2 * dx)))

            fin_function = Function(
                Q, name='RecFin').assign(
                    T.block_variable.checkpoint.restore())
            init_function = Function(
                Q1, name='RecInit').assign(
                    Tic.block_variable.checkpoint.restore())

            # Writing out the final function
            checkpoint_data = CheckpointFile(
                f"./solutions/it_{self.iteration}.h5", "w")
            checkpoint_data.save_mesh(mesh)
            checkpoint_data.save_function(fin_function)
            checkpoint_data.save_function(init_function)
            checkpoint_data.close()

            # updating iteration number
            self.iteration += 1

    # Set up bounds, which will later be used to
    # enforce boundary conditions in inversion:
    T_lb = Function(Tic.function_space(), name="LB_Temperature")
    T_ub = Function(Tic.function_space(), name="UB_Temperature")
    T_lb.assign(0.0)
    T_ub.assign(1.0)

    # Optimise using ROL - note when doing Taylor test this can be turned off:
    minp = MinimizationProblem(reduced_functional, bounds=(T_lb, T_ub))

    # Checkpointing, setting a mesh and directory for checkpoints
    ROLCheckpointManager.set_mesh('./mesh.h5', mesh.name)
    ROLCheckpointManager.set_checkpoint_dir('checkpoints')

    lin_more = LinMoreOptimiser(
        minp,
        minimisation_parameters,
        callback=ROL_callback())

    # This is in case we want to set a restart point
    # if my_restarter.iteration > 0:
    #    lin_more.reload(my_restarter.iteration)

    lin_more.run()

    # make sure we keep annotating after this
    continue_annotation()


if __name__ == "__main__":
    main()
