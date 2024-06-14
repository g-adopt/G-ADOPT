from firedrake import (
    Constant, DirichletBC, FacetNormal, Function,
    assemble, dot, ds, dx, grad, norm, sqrt,
)
from firedrake.ufl_expr import extract_unique_domain
from mpi4py import MPI

from .utility import CombinedSurfaceMeasure


class GeodynamicalDiagnostics:
    """Typical simulation diagnostics used in geodynamical simulations.

    Arguments:
      z:         Firedrake function for the mixed Stokes function space
      T:         Firedrake function for the temperature
      bottom_id: bottom boundary identifier
      top_id:    top boundary identifier
      degree:    degree of the polynomial approximation

    Note:
      All the diagnostics are returned as a float value.

    Functions:
      u_rms: Root-mean squared velocity
      u_rms_top: Root-mean squared velocity along the top boundary
      Nu_top: Nusselt number at the top boundary
      Nu_bottom: Nusselt number at the bottom boundary
      T_avg: Average temperature in the domain
      ux_max: Maximum velocity (optionally over a given boundary)

    """

    def __init__(
        self,
        z: Function,
        T: Function,
        bottom_id: int,
        top_id: int,
        degree: int = 4,
    ):
        mesh = extract_unique_domain(z)

        self.u, self.p, *_ = z.subfunctions
        self.T = T

        self.dx = dx(domain=mesh, degree=degree)
        self.ds = (
            CombinedSurfaceMeasure(mesh, degree)
            if T.function_space().extruded
            else ds(mesh)
        )
        self.ds_t = self.ds(top_id)
        self.ds_b = self.ds(bottom_id)

        self.n = FacetNormal(mesh)

        self.domain_volume = assemble(Constant(1) * self.dx)
        self.top_surface = assemble(Constant(1) * self.ds_t)
        self.bottom_surface = assemble(Constant(1) * self.ds_b)

    def u_rms(self):
        return norm(self.u) / sqrt(self.domain_volume)

    def u_rms_top(self) -> float:
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -assemble(dot(grad(self.T), self.n) * self.ds_t) / self.top_surface

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b) / self.bottom_surface

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume

    def ux_max(self, boundary_id=None) -> float:
        ux_data = self.u.dat.data_ro_with_halos[:, 0]

        if boundary_id:
            bcu = DirichletBC(self.u.function_space(), 0, boundary_id)
            ux_data = ux_data[bcu.nodes]

        return self.u.comm.allreduce(ux_data.max(initial=0), MPI.MAX)
