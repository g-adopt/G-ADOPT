"""Compositional benchmark.
Schmeling, H., Babeyko, A. Y., Enns, A., Faccenna, C., Funiciello, F., Gerya, T.,
... & Van Hunen, J. (2008).
A benchmark comparison of spontaneous subduction models—Towards a free surface.
Physics of the Earth and Planetary Interiors, 171(1-4), 198-223.
"""

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
from pandas import read_excel

from gadopt.level_set_tools import min_max_height

from .materials import air, lithosphere, mantle


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    height = min_max_height(
        diag_vars["level_set"][1], diag_vars["epsilon"], side=1, mode="min"
    )

    diag_fields["output_time"].append(simu_time / 8.64e4 / 365.25 / 1e6)
    diag_fields["slab_tip_depth"].append((domain_dims[1] - 5e4 - height) / 1e3)

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        datafile = "data/zslab-case1-best-reformatted.xlsx"

        model_names_schmeling = read_excel(
            datafile, sheet_name="Tabelle1", header=None, skiprows=2, nrows=1
        )

        geom_cols = []
        geom_col_names = []
        for col in model_names_schmeling:
            model_name = model_names_schmeling[col].item()
            if isinstance(model_name, str) and "geom" in model_name:
                geom_cols.append(col)
                geom_col_names.append(model_name)

        cols_to_read = np.repeat(geom_cols, 2)
        cols_to_read[1::2] += 1

        model_data = read_excel(
            datafile,
            sheet_name="Tabelle1",
            header=None,
            usecols=cols_to_read,
            skiprows=4,
        )

        fdcon_data = read_excel(
            datafile, sheet_name="zslabmodel5", header=None, usecols=[1, 2]
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)

        ax.grid()

        ax.invert_yaxis()
        ax.set_xlim(right=65)

        ax.set_xlabel("Time (Myr)")
        ax.set_ylabel("Slab tip depth (km)")

        ax.plot(
            fdcon_data[1].dropna().values,
            fdcon_data[2].dropna().values,
            linestyle="dotted",
            label="FDCON 561x141 geom",
        )

        for col, col_name in zip(geom_cols, geom_col_names):
            ax.plot(
                model_data[col].dropna().values,
                model_data[col + 1].dropna().values,
                linestyle="dotted",
                label=col_name,
            )

        ax.plot(
            diag_fields["output_time"],
            diag_fields["slab_tip_depth"],
            color="black",
            label="Conservative level set",
        )

        ax.legend()

        fig.savefig(
            f"{output_path}/slab_tip_depth_{tag}.pdf", dpi=300, bbox_inches="tight"
        )


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (3e6, 7.5e5)
mesh_gen = "gmsh"

# Parameters to initialise surface level set
interface_coords_x = np.array([0.0, domain_dims[0]])
callable_args = (surface_slope := 0, surface_coord_y := 7e5)
surface_signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": "line",
    "interface_args": (interface_coords_x, *callable_args),
}
# Parameters to initialise slab level set
slab_interface_coords = [
    (domain_dims[0], 7e5),
    (1e6, 7e5),
    (1e6, 5e5),
    (1.1e6, 5e5),
    (1.1e6, 6e5),
    (domain_dims[0], 6e5),
]
slab_boundary_coords = [(domain_dims[0], 7e5)]
slab_signed_distance_kwargs = {
    "interface_geometry": "polygon",
    "interface_coordinates": slab_interface_coords,
    "boundary_coordinates": slab_boundary_coords,
}
# The following list must be ordered such that, unpacking from the end, each dictionary
# contains the keyword arguments required to initialise the signed-distance array
# corresponding to the interface between a given material and the remainder of the
# numerical domain (all previous materials excluded). By convention, the material thus
# isolated occupies the positive side of the signed-distance array.
signed_distance_kwargs_list = [
    surface_signed_distance_kwargs,
    slab_signed_distance_kwargs,
]

# Material ordering must follow the logic implemented in the above list. In other words,
# the last material in the below list must correspond to the portion of the numerical
# domain isolated by the signed-distance array calculated using the last dictionary in
# the above list. The first material in the below list will, therefore, occupy the
# negative side of the signed-distance array calculated from the first dictionary above.
materials = [mantle, air, lithosphere]

# Approximation parameters
dimensional = True
g = 9.81

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e11
dump_period = 8e5 * 365.25 * 8.64e4
checkpoint_period = 5
time_end = 6e7 * 365.25 * 8.64e4

# Diagnostic objects
diag_fields = {"output_time": [], "slab_tip_depth": []}
