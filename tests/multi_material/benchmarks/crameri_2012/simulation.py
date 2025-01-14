"""Compositional benchmark.
Crameri, F., Schmeling, H., Golabek, G. J., Duretz, T., Orendt, R.,
Buiter, S. J. H., ... & Tackley, P. J. (2012).
A comparison of numerical surface topography calculations in geodynamic modelling:
an evaluation of the 'sticky air' method.
Geophysical Journal International, 189(1), 38-54.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from gadopt.level_set_tools import min_max_height

from .materials import air, lithosphere, mantle


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    max_topo = min_max_height(
        diag_vars["level_set"][1], diag_vars["epsilon"], side=0, mode="max"
    )
    max_topo_analytical = surface_deflection / 1e3 * np.exp(relaxation_rate * simu_time)

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e3)
    diag_fields["max_topography"].append((max_topo - surface_coord_y) / 1e3)
    diag_fields["max_topography_analytical"].append(max_topo_analytical)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.set_xlabel("Time (kyr)")
        ax.set_ylabel("Maximum topography (km)")

        ax.plot(
            diag_fields["output_time"],
            diag_fields["max_topography_analytical"],
            label="Analytical (Crameri et al., 2012)",
        )
        ax.plot(
            diag_fields["output_time"],
            diag_fields["max_topography"],
            label="Conservative level set",
        )

        ax.legend()

        fig.savefig(
            f"{output_path}/maximum_topography_{tag}.pdf", dpi=300, bbox_inches="tight"
        )


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (2.8e6, 8e5)
mesh_gen = "gmsh"

# Parameters to initialise surface level set
interface_coords_x = np.linspace(0.0, domain_dims[0], int(domain_dims[0] / 1e3) + 1)
callable_args = (
    surface_deflection := 7e3,
    surface_perturbation_wavelength := domain_dims[0],
    surface_coord_y := 7e5,
)
surface_signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": "cosine",
    "interface_args": (interface_coords_x, *callable_args),
}
# Parameters to initialise LAB level set
interface_coords_x = np.array([0.0, domain_dims[0]])
callable_args = (lab_slope := 0, lab_coord_y := 6e5)
lab_signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": "line",
    "interface_args": (interface_coords_x, *callable_args),
}
# The following list must be ordered such that, unpacking from the end, each dictionary
# contains the keyword arguments required to initialise the signed-distance array
# corresponding to the interface between a given material and the remainder of the
# numerical domain (all previous materials excluded). By convention, the material thus
# isolated occupies the positive side of the signed-distance array.
signed_distance_kwargs_list = [
    lab_signed_distance_kwargs,
    surface_signed_distance_kwargs,
]

# Material ordering must follow the logic implemented in the above list. In other words,
# the last material in the below list must correspond to the portion of the numerical
# domain isolated by the signed-distance array calculated using the last dictionary in
# the above list. The first material in the below list will, therefore, occupy the
# negative side of the signed-distance array calculated from the first dictionary above.
materials = [mantle, lithosphere, air]

# Approximation parameters
dimensional = True
g = 10

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"u": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e10
dump_period = 2e3 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 1e5 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"output_time": [], "max_topography": [], "max_topography_analytical": []}
relaxation_rate = -0.2139e-11
