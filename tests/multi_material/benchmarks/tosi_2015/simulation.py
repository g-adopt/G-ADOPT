"""Thermal benchmark.
Tosi, N., Stein, C., Noack, L., Hüttig, C., Maierova, P., Samuel, H., ...
& Tackley, P. J. (2015).
A community benchmark for viscoplastic thermal convection in a 2-D square box.
Geochemistry, Geophysics, Geosystems, 16(7), 2175-2196.
"""

import firedrake as fd
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from .materials import material


def initialise_temperature(temperature):
    mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

    temperature.interpolate(
        1
        - mesh_coords[1]
        + A * fd.cos(fd.pi * mesh_coords[0]) * fd.sin(fd.pi * mesh_coords[1])
    )


def diagnostics(simu_time, geo_diag, diag_vars, output_path):
    diag_fields["output_time"].append(simu_time)
    diag_fields["avg_temperature"].append(geo_diag.T_avg())
    diag_fields["nusselt_top"].append(geo_diag.Nu_top())
    diag_fields["nusselt_bottom"].append(geo_diag.Nu_bottom())
    diag_fields["rms_velocity"].append(geo_diag.u_rms())
    diag_fields["min_visc"].append(
        geo_diag.T.comm.allreduce(diag_vars["viscosity"].dat.data.min(), MPI.MIN)
    )
    diag_fields["max_visc"].append(
        geo_diag.T.comm.allreduce(diag_vars["viscosity"].dat.data.max(), MPI.MAX)
    )

    if MPI.COMM_WORLD.rank == 0:
        np.savez(
            f"{output_path}/output_{checkpoint_restart}_{tag}", diag_fields=diag_fields
        )


def plot_diagnostics(output_path):
    if MPI.COMM_WORLD.rank == 0:
        fig, ax = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

        for axis in ax.flatten():
            axis.grid()
            axis.set_xlabel("Time (non-dimensional)")

        ax[0, 0].set_ylabel("Average temperature (non-dimensional)")
        ax[0, 1].set_ylabel("Top Nusselt number (non-dimensional)")
        ax[0, 2].set_ylabel("Bottom Nusselt number (non-dimensional)")
        ax[1, 0].set_ylabel("Root-mean-square velocity (non-dimensional)")
        ax[1, 1].set_ylabel("Minimum viscosity (non-dimensional)")
        ax[1, 2].set_ylabel("Maximum viscosity (non-dimensional)")

        ax[0, 0].plot(diag_fields["output_time"], diag_fields["avg_temperature"])
        ax[0, 1].plot(diag_fields["output_time"], diag_fields["nusselt_top"])
        ax[0, 2].plot(diag_fields["output_time"], diag_fields["nusselt_bottom"])
        ax[1, 0].plot(diag_fields["output_time"], diag_fields["rms_velocity"])
        ax[1, 1].plot(diag_fields["output_time"], diag_fields["min_visc"])
        ax[1, 2].plot(diag_fields["output_time"], diag_fields["max_visc"])

        fig.savefig(
            f"{output_path}/diagnostics_{tag}.pdf", dpi=300, bbox_inches="tight"
        )


# A simulation name tag
tag = "reference"
# 0 indicates the initial run and positive integers corresponding restart runs.
checkpoint_restart = 0

# Mesh resolution should be sufficient to capture eventual small-scale dynamics
# in the neighbourhood of material interfaces tracked by the level-set approach.
# Insufficient mesh refinement can lead to unwanted motion of material interfaces.
domain_dims = (1, 1)
mesh_gen = "firedrake"
mesh_elements = (48, 48)

# Parameters to initialise level set
interface_coords_x = np.array([0.0, domain_dims[0]])
callable_args = (interface_slope := 0, interface_coord_y := 0.5)
signed_distance_kwargs = {
    "interface_geometry": "curve",
    "interface_callable": "line",
    "interface_args": (interface_coords_x, *callable_args),
}
# The following list must be ordered such that, unpacking from the end, each dictionary
# contains the keyword arguments required to initialise the signed-distance array
# corresponding to the interface between a given material and the remainder of the
# numerical domain (all previous materials excluded). By convention, the material thus
# isolated occupies the positive side of the signed-distance array.
signed_distance_kwargs_list = [signed_distance_kwargs]

# Material ordering must follow the logic implemented in the above list. In other words,
# the last material in the below list must correspond to the portion of the numerical
# domain isolated by the signed-distance array calculated using the last dictionary in
# the above list. The first material in the below list will, therefore, occupy the
# negative side of the signed-distance array calculated from the first dictionary above.
materials = [material, material]

# Approximation parameters
dimensional = False
Ra = 1e2

# Parameters to initialise temperature
A = 0.01

# Boundary conditions with mapping {1: left, 2: right, 3: bottom, 4: top}
temp_bcs = {3: {"T": 1}, 4: {"T": 0}}
stokes_bcs = {1: {"ux": 0}, 2: {"ux": 0}, 3: {"uy": 0}, 4: {"uy": 0}}

# Timestepping objects
initial_timestep = 1e-6
dump_period = 1e-3
checkpoint_period = 5
steady_state_threshold = 1e-5

# Diagnostic objects
diag_fields = {
    "output_time": [],
    "avg_temperature": [],
    "nusselt_top": [],
    "nusselt_bottom": [],
    "rms_velocity": [],
    "min_visc": [],
    "max_visc": [],
}
