import firedrake as fd


def mu(velocity, temperature):
    mesh_coords = fd.SpatialCoordinate(temperature.function_space().mesh())

    strain_rate = fd.sym(fd.grad(velocity))

    visc_lin = fd.exp(
        -fd.ln(visc_contrast_temp) * temperature
        + fd.ln(visc_contrast_pres) * (1 - mesh_coords[1])
    )
    visc_plast = visc_eff_high_stress + yield_stress / fd.sqrt(
        fd.inner(strain_rate, strain_rate) + 1e-99
    )

    return 2 / (1 / visc_lin + 1 / visc_plast)


Ra_c = 0

visc_contrast_temp = 1e5
visc_contrast_pres = 1e1
visc_eff_high_stress = 1e-3
yield_stress = 1
