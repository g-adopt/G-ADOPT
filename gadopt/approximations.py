r"""This module provides classes that emulate physical approximations of fluid dynamics
systems by exposing methods to calculate specific terms in the corresponding
mathematical equations. Users instantiate the appropriate class by providing relevant
parameters and pass the instance to other objects, such as solvers. Under the hood,
G-ADOPT queries variables and methods from the approximation.

"""

import abc
from numbers import Number
from typing import Optional

from firedrake import Function, Identity, div, grad, inner, sym, ufl

from .utility import ensure_constant, vertical_component

__all__ = [
    "BoussinesqApproximation",
    "ExtendedBoussinesqApproximation",
    "TruncatedAnelasticLiquidApproximation",
    "AnelasticLiquidApproximation"
]


class BaseApproximation(abc.ABC):
    """Base class to provide expressions for the coupled Stokes and Energy system.

    The basic assumption is that we are solving (to be extended when needed)

        div(dev_stress) + grad p + buoyancy(T, p) * khat = 0
        div(rho_continuity * u) = 0
        rhocp DT/Dt + linearized_energy_sink(u) * T
          = div(kappa * grad(Tbar + T)) + energy_source(u)

    where the following terms are provided by Approximation methods:

    - linearized_energy_sink(u) = 0 (BA), Di * rhobar * alphabar * g * w (EBA),
      or Di * rhobar * alphabar * w (TALA/ALA)
    - kappa() is diffusivity or conductivity depending on rhocp()
    - Tbar is 0 or reference temperature profile (ALA)
    - dev_stress depends on the compressible property (False or True):
        - if compressible then dev_stress = mu * [sym(grad(u) - 2/3 div(u)]
        - if not compressible then dev_stress = mu * sym(grad(u)) and
          rho_continuity is assumed to be 1

    """

    @property
    @abc.abstractmethod
    def compressible(self) -> bool:
        """Defines compressibility.

        Returns:
          A boolean signalling if the governing equations are in compressible form.

        """
        pass

    @abc.abstractmethod
    def stress(self, u: Function) -> ufl.core.expr.Expr:
        """Defines the deviatoric stress.

        Returns:
          A UFL expression for the deviatoric stress.

        """
        pass

    @abc.abstractmethod
    def buoyancy(self, p: Function, T: Function) -> ufl.core.expr.Expr:
        """Defines the buoyancy force.

        Returns:
          A UFL expression for the buoyancy term (momentum source in gravity direction).

        """
        pass

    @abc.abstractmethod
    def rho_continuity(self) -> ufl.core.expr.Expr:
        """Defines density.

        Returns:
          A UFL expression for density in the mass continuity equation.

        """
        pass

    @abc.abstractmethod
    def rhocp(self) -> ufl.core.expr.Expr:
        """Defines the volumetric heat capacity.

        Returns:
          A UFL expression for the volumetric heat capacity in the energy equation.

        """
        pass

    @abc.abstractmethod
    def kappa(self) -> ufl.core.expr.Expr:
        """Defines thermal diffusivity.

        Returns:
          A UFL expression for thermal diffusivity.

        """
        pass

    @property
    @abc.abstractmethod
    def Tbar(self) -> Function:
        """Defines the reference temperature profile.

        Returns:
          A Firedrake function for the reference temperature profile.

        """
        pass

    @abc.abstractmethod
    def linearized_energy_sink(self, u) -> ufl.core.expr.Expr:
        """Defines temperature-related sink terms.

        Returns:
          A UFL expression for temperature-related sink terms in the energy equation.

        """
        pass

    @abc.abstractmethod
    def energy_source(self, u) -> ufl.core.expr.Expr:
        """Defines additional terms.

        Returns:
          A UFL expression for additional independent terms in the energy equation.

        """
        pass

    @abc.abstractmethod
    def free_surface_terms(self, p, T, eta, theta_fs) -> tuple[ufl.core.expr.Expr]:
        """Defines free surface normal stress in the momentum equation and prefactor
        multiplying the free surface equation.

        The normal stress depends on the density contrast across the free surface.
        Depending on the `variable_rho_fs` argument, this contrast either involves the
        interior reference density or the full interior density that accounts for
        changes in temperature and composition within the domain. For dimensional
        simulations, the user should specify `delta_rho_fs`, defined as the difference
        between the reference interior density and the exterior density. For
        non-dimensional simulations, the user should specify `RaFS`, the equivalent of
        the Rayleigh number for the density contrast across the free surface.

        The prefactor multiplies the free surface equation to ensure the top-right and
        bottom-left corners of the block matrix remain symmetric. This is similar to
        rescaling eta -> eta_tilde in Kramer et al. (2012, see block matrix shown in
        Eq. 23).

        Returns:
          A UFL expression for the free surface normal stress and a UFL expression for
          the free surface equation prefactor.

        """
        pass


class BoussinesqApproximation(BaseApproximation):
    """Expressions for the Boussinesq approximation.

    Density variations are considered small and only affect the buoyancy term. Reference
    parameters are typically constant. Viscous dissipation is neglected (Di << 1).

    Arguments:
      Ra:        Rayleigh number
      mu:        dynamic viscosity
      rho:       reference density
      alpha:     coefficient of thermal expansion
      T0:        reference temperature
      g:         gravitational acceleration
      Ra_c:       compositional Rayleigh number; product of the Rayleigh and buoyancy numbers
      delta_rho: compositional density difference from the reference density
      kappa:     thermal diffusivity
      H:         internal heating rate

    Note:
      The thermal diffusivity, gravitational acceleration, reference
      density, and coefficient of thermal expansion are normally kept
      at 1 when non-dimensionalised.

    """
    compressible = False
    Tbar = 0

    def __init__(
        self,
        Ra: Function | Number,
        *,
        mu: Function | Number = 1,
        rho: Function | Number = 1,
        alpha: Function | Number = 1,
        T0: Function | Number = 0,
        g: Function | Number = 1,
        Ra_c: Function | Number = 0,
        delta_rho: Function | Number = 1,
        kappa: Function | Number = 1,
        H: Function | Number = 0,
    ):
        self.Ra = ensure_constant(Ra)
        self.mu = ensure_constant(mu)
        self.rho = ensure_constant(rho)
        self.alpha = ensure_constant(alpha)
        self.T0 = T0
        self.g = ensure_constant(g)
        self.kappa_ref = ensure_constant(kappa)
        self.Ra_c = Ra_c
        self.delta_rho = ensure_constant(delta_rho)
        self.H = ensure_constant(H)

    def stress(self, u):
        return 2 * self.mu * sym(grad(u))

    def buoyancy(self, p, T):
        return (
            self.Ra * self.rho * self.alpha * (T - self.T0) * self.g
            - self.Ra_c * self.delta_rho * self.g
        )

    def rho_continuity(self):
        return 1

    def rhocp(self):
        return 1

    def kappa(self):
        return self.kappa_ref

    def linearized_energy_sink(self, u):
        return 0

    def energy_source(self, u):
        return self.rho * self.H

    def free_surface_terms(
        self, p, T, eta, theta_fs, *, variable_rho_fs=True, RaFS=1, delta_rho_fs=1
    ):
        free_surface_normal_stress = RaFS * delta_rho_fs * self.g * eta
        if variable_rho_fs:
            free_surface_normal_stress -= self.buoyancy(p, T) * eta

        prefactor = -theta_fs * RaFS * delta_rho_fs * self.g

        return free_surface_normal_stress, prefactor


class ExtendedBoussinesqApproximation(BoussinesqApproximation):
    """Expressions for the extended Boussinesq approximation.

    Extends the Boussinesq approximation by including viscous dissipation and work
    against gravity (both scaled with Di).

    Arguments:
      Ra: Rayleigh number
      Di: Dissipation number
      mu: dynamic viscosity
      H:  volumetric heat production

    Other Arguments:
      rho (Number):           reference density
      alpha (Number):         coefficient of thermal expansion
      T0 (Function | Number): reference temperature
      g (Number):             gravitational acceleration
      Ra_c (Number):           compositional Rayleigh number; product
                              of the Rayleigh and buoyancy numbers
      delta_rho (Number):     compositional density difference from
                              the reference density
      kappa (Number):         thermal diffusivity

    Note:
      The thermal diffusivity, gravitational acceleration, reference
      density, and coefficient of thermal expansion are normally kept
      at 1 when non-dimensionalised.

    """
    compressible = False

    def __init__(self, Ra: Number, Di: Number, *, H: Optional[Number] = None, **kwargs):
        super().__init__(Ra, **kwargs)
        self.Di = Di
        self.H = H

    def viscous_dissipation(self, u):
        phi = inner(self.stress(u), grad(u))
        return phi * self.Di / self.Ra

    def linearized_energy_sink(self, u):
        w = vertical_component(u)
        return self.Di * self.alpha * self.rho * self.g * w

    def work_against_gravity(self, u, T):
        return self.linearized_energy_sink(u) * T

    def energy_source(self, u):
        source = self.viscous_dissipation(u)
        if self.H:
            source += self.H * self.rho
        return source


class TruncatedAnelasticLiquidApproximation(ExtendedBoussinesqApproximation):
    """Truncated Anelastic Liquid Approximation

    Compressible approximation. Excludes linear dependence of density on pressure.

    Arguments:
      Ra:     Rayleigh number
      Di:     Dissipation number
      Tbar:   reference temperature. In the diffusion term we use Tbar + T (i.e. T is the pertubartion)
      cp:     reference specific heat at constant pressure

    Other Arguments:
      rho (Number):           reference density
      alpha (Number):         reference thermal expansion coefficient
      T0 (Function | Number): reference temperature
      g (Number):             gravitational acceleration
      Ra_c (Number):           compositional Rayleigh number; product
                              of the Rayleigh and buoyancy numbers
      delta_rho (Number):     compositional density difference from
                              the reference density
      kappa (Number):         diffusivity
      mu (Number):            viscosity used in viscous dissipation
      H (Number):             volumetric heat production

    Note:
      Other keyword arguments may be depth-dependent, but default to 1 if not supplied.

    """
    compressible = True

    def __init__(self,
                 Ra: Number,
                 Di: Number,
                 *,
                 Tbar: Function | Number = 0,
                 cp: Function | Number = 1,
                 **kwargs):
        super().__init__(Ra, Di, **kwargs)
        self.Tbar = Tbar
        self.cp = cp

    def stress(self, u):
        stress = super().stress(u)
        dim = len(u)  # Geometric dimension, i.e. 2D or 3D
        return stress - 2/3 * self.mu * Identity(dim) * div(u)

    def rho_continuity(self):
        return self.rho

    def rhocp(self):
        return self.rho * self.cp


class AnelasticLiquidApproximation(TruncatedAnelasticLiquidApproximation):
    """Anelastic Liquid Approximation

    Compressible approximation. Includes linear dependence of density on pressure.

    Arguments:
      Ra:     Rayleigh number
      Di:     Dissipation number
      chi:    reference isothermal compressibility
      gamma0: Gruneisen number (in pressure-dependent buoyancy term)
      cp0:    specific heat at constant *pressure*, reference for entire Mantle (in pressure-dependent buoyancy term)
      cv0:    specific heat at constant *volume*, reference for entire Mantle (in pressure-dependent buoyancy term)

    Other Arguments:
      rho (Number):           reference density
      alpha (Number):         reference thermal expansion coefficient
      T0 (Function | Number): reference temperature
      g (Number):             gravitational acceleration
      Ra_c (Number):           compositional Rayleigh number; product
                              of the Rayleigh and buoyancy numbers
      delta_rho (Number):     compositional density difference from
                              the reference density
      kappa (Number):         diffusivity
      mu (Number):            viscosity used in viscous dissipation
      H (Number):             volumetric heat production
      Tbar (Number):          reference temperature. In the diffusion
                              term we use Tbar + T (i.e. T is the pertubartion)
      cp (Number):            reference specific heat at constant pressure

    """

    def __init__(self,
                 Ra: Number,
                 Di: Number,
                 *,
                 chi: Function | Number = 1,
                 gamma0: Function | Number = 1,
                 cp0: Function | Number = 1,
                 cv0: Function | Number = 1,
                 **kwargs):
        super().__init__(Ra, Di, **kwargs)
        # Dynamic pressure contribution towards buoyancy
        self.chi = chi
        self.gamma0, self.cp0, self.cv0 = gamma0, cp0, cv0

    def dbuoyancydp(self, p, T: ufl.core.expr.Expr):
        return -self.Di * self.cp0 / self.cv0 / self.gamma0 * self.g * self.rho * self.chi

    def buoyancy(self, p, T):
        pressure_part = self.dbuoyancydp(p, T) * p
        temperature_part = super().buoyancy(p, T)
        return pressure_part + temperature_part


class SmallDisplacementViscoelasticApproximation():
    """Expressions for the small displacement viscoelastic approximation.

    By assuming a small displacement, we can linearise the problem, assuming a perturbation
    away from a reference state. We assume a Maxwell viscoelastic rheology, i.e.
    stress is the same but viscous and elastic strains combine linearly. We follow
    the approach by Zhong et al. 2003 redefining the problem in terms of incremental
    displacement, i.e. velocity * dt where dt is the timestep. This produces a mixed
    stokes system for incremental displacement and pressure which can be solved in
    the same way as mantle convection (where unknowns are velocity and pressure),
    with a modfied viscosity and stress term accounting for the deviatoric stress
    at the previous timestep.

    Zhong, Shijie, Archie Paulson, and John Wahr.
    "Three-dimensional finite-element modelling of Earth’s viscoelastic
    deformation: effects of lateral variations lithospheric thickness."
    Geophysical Journal International 155.2 (2003): 679-695.

    N.b. that the implentation currently assumes all terms are dimensional.

    Arguments:
      density:       background density
      shear_modulus: shear modulus
      viscosity:     viscosity
      g:             gravitational acceleration

    """
    compressible = False

    def __init__(
        self,
        density: Function | Number,
        shear_modulus: Function | Number,
        viscosity: Function | Number,
        *,
        g: Function | Number = 1,
    ):

        self.density = ensure_constant(density)
        self.shear_modulus = ensure_constant(shear_modulus)
        self.viscosity = ensure_constant(viscosity)
        self.g = ensure_constant(g)

        self.maxwell_time = viscosity / shear_modulus

    def effective_viscosity(self, dt):
        return self.viscosity / (self.maxwell_time + dt / 2)

    def prefactor_prestress(self, dt):
        return (self.maxwell_time - dt / 2) / (self.maxwell_time + dt / 2)

    def stress(self, u, stress_old, dt):
        return 2 * self.effective_viscosity(dt) * sym(grad(u)) + stress_old

    def buoyancy(self, displacement):
        # Buoyancy term rho1, coming from linearisation and integrating the continuity equation w.r.t time
        # accounts for advection of density in the absence of an evolution equation for temperature
        return -self.g * -inner(displacement, grad(self.density))

    def free_surface_terms(self, p, T, eta, theta_fs, *, delta_rho_fs=1):
        free_surface_normal_stress = delta_rho_fs * self.g * eta
        # prefactor only needed when solving eta as part of mixed system
        return free_surface_normal_stress, None

    def rho_continuity(self):
        return 1
