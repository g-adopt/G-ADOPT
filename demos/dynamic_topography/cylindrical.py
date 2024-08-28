"""
Test case for normal stresses on a sphere
"""

from gadopt import *

newton_stokes_solver_parameters = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "l2",
    "snes_max_it": 100,
    "snes_atol": 1e-10,
    "snes_rtol": 1e-5,
    "snes_stol": 0,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
    "snes_converged_reason": None,
    "fieldsplit_0": {
        "ksp_converged_reason": None,
    },
    "fieldsplit_1": {
        "ksp_converged_reason": None,
    },
}


# Start with a previously-initialised temperature field
with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    mesh = f.load_mesh("firedrake_default_extruded")

# Set up function spaces for the Q2Q1 pair
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Z = MixedFunctionSpace([V, W])

# Test functions and functions to hold solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

Ra = Constant(1e7)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

with CheckpointFile("../adjoint_2d_cylindrical/Checkpoint230.h5", mode="r") as f:
    T = f.load_function(mesh, "Temperature")

Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=True)
Z_near_nullspace = create_stokes_nullspace(
    Z, closed=False, rotational=True, translations=[0, 1]
)

# Free-slip velocity boundary condition on all sides
stokes_bcs = {
    "bottom": {"un": 0},
    "top": {"un": 0},
}

stokes_solver = StokesSolver(
    z,
    T,
    approximation,
    bcs=stokes_bcs,
    cartesian=False,
    nullspace=Z_nullspace,
    transpose_nullspace=Z_nullspace,
    near_nullspace=Z_near_nullspace,
    solver_parameters=newton_stokes_solver_parameters,
)

# Split and rename the velocity and pressure functions
# so that they can be used for visualisation
u, p = z.subfunctions
u.rename("Velocity")
p.rename("Pressure")

stokes_solver.solve()
force = Function(W, name="force")
stokes_solver.deviatoric_normal_stress(force, "top")

# # Create output file and select output_frequency
output_file = VTKFile("output.pvd")
output_file.write(u, p, T, force)
