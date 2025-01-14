import firedrake as fd


def mu(velocity, *args):
    strain_rate = fd.sym(fd.grad(velocity))
    strain_rate_sec_inv = fd.sqrt(fd.inner(strain_rate, strain_rate) / 2 + 1e-99)

    viscosity = visc_coeff * strain_rate_sec_inv ** (1 / stress_exponent - 1)

    return fd.min_value(fd.max_value(viscosity, visc_bounds[0]), visc_bounds[1])


rho = 3300

visc_coeff = 4.75e11
stress_exponent = 4.0
visc_bounds = (1e21, 1e25)
