from firedrake import assemble, Constant, dx, ds, sqrt, dot, grad, FacetNormal


def domain_volume(mesh):
    return assemble(Constant(1)*dx(domain=mesh))


class GeodynamicalDiagnostics:

    def __init__(self, u, p, T, bottom_id, top_id, degree=4):
        mesh = u.ufl_domain()
        self.domain_volume = domain_volume(mesh)
        self.u = u
        self.p = p
        self.T = T
        self.dx = dx(degree=degree)
        self.ds_t = ds(top_id, degree=degree)
        self.ds_b = ds(bottom_id, degree=degree)
        self.n = FacetNormal(mesh)

    def u_rms(self):
        return sqrt(assemble(dot(self.u, self.u) * self.dx)) * sqrt(1./self.domain_volume)

    def u_rms_top(self):
        return sqrt(assemble(dot(self.u, self.u) * self.ds_t))

    def Nu_top(self):
        return -1 * assemble(dot(grad(self.T), self.n) * self.ds_t)

    def Nu_bottom(self):
        return assemble(dot(grad(self.T), self.n) * self.ds_b)

    def T_avg(self):
        return assemble(self.T * self.dx) / self.domain_volume