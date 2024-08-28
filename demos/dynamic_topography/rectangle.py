from gadopt import *

# Set up geometry:
nx, ny = 5, 5
mesh = UnitSquareMesh(nx, ny, quadrilateral=True)  # Square mesh generated via firedrake
left_id, right_id, bottom_id, top_id = 1, 2, 3, 4  # Boundary IDs

# Set up function spaces - currently using the bilinear Q2Q1 element pair:
V = VectorFunctionSpace(mesh, "CG", 2)  # Velocity function space (vector)
W = FunctionSpace(mesh, "CG", 1)  # Pressure function space (scalar)
Q = FunctionSpace(mesh, "CG", 2)  # Temperature function space (scalar)
Z = MixedFunctionSpace([V, W])  # Mixed function space.

# Function to store the solutions:
z = Function(Z)  # a field over the mixed function space Z.
u, p = split(z)  # Returns symbolic UFL expression for u and p

# Set up temperature field and initialise:
X = SpatialCoordinate(mesh)
T = Function(Q, name="Temperature")
T.interpolate((1.0-X[1]) + (0.1*cos(pi*X[0])*sin(pi*X[1])))

delta_t = Constant(1e-6)  # Initial time-step

# Stokes related constants (note that since these are included in UFL, they are wrapped inside Constant):
Ra = Constant(1e4)  # Rayleigh number
approximation = BoussinesqApproximation(Ra)

# Nullspaces and near-nullspaces:
Z_nullspace = create_stokes_nullspace(Z, closed=True, rotational=False)

# Write output files in VTK format:
u, p = z.subfunctions  # Do this first to extract individual velocity and pressure fields.
# Next rename for output:
u.rename("Velocity")
p.rename("Pressure")

stokes_bcs = {
    bottom_id: {'uy': 0},
    top_id: {'uy': 0},
    left_id: {'ux': 0},
    right_id: {'ux': 0},
}

stokes_solver = StokesSolver(z, T, approximation, bcs=stokes_bcs,
                             cartesian=True, constant_jacobian=True,
                             nullspace=Z_nullspace, transpose_nullspace=Z_nullspace)

# Solve Stokes sytem:
stokes_solver.solve()
force = Function(W, name="force")
stokes_solver.deviatoric_normal_stress(force, 4)

# Create output file and select output_frequency:
output_file = VTKFile("output.pvd")
output_file.write(T, force, p)
