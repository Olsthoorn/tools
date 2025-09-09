# Soil base Class

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

from functools import cached_property

from scipy.stats import norm
from scipy.special import gammaln, erfc
from scipy.stats import gamma
from scipy.interpolate import PchipInterpolator
from typing import Callable


class SoilBase(ABC):
    """Abstract base class for soil hydraulic properties.
    
    Note that S is used for reduced saturation where capitial theta is used in most literature.    
        Theta = S = (theta - theta_r) / (theta_s - theta_r)
    
    and that saturation is sat (not S) defined as
        sat = theta / theta_s    
    """

    def __init__(self, soil_code: str) -> None:
        if not hasattr(self.__class__, 'data'):
            raise AttributeError("Load soils first with Soil.load(filename)")
        
        if soil_code not in self.__class__.data.index:
            raise KeyError(f"Soil code {soil_code} not found in database.")
                
        self.props = self.__class__.data.loc[soil_code].copy()
            
        return None
        
    # --- Required methods for subclasses ---
    @classmethod
    @abstractmethod
    def load_soils(cls, wbook: str | Path) -> None:
        """Load soils from workbook."""
        pass
    
    @classmethod
    def pretty_data(cls) -> None:
        if cls.data is None:
            return "No soil  data loaded."
        print(cls.data.to_string()) # nicely formatted DataFrame

    def __str__(self) -> str:
        """Print soil properties in a readable format, starting with Hoofdsoort and Omschrijving."""
        
        lines = [f"Grondsoort code: {self.code}"]
        for k in self.props.index:
            val = self.props[k]
            unit = super.dimensions.get(k, '')
            lines.append(f"{k:12} {unit:6}: {val}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Soil('{self.props['code']}')"
    
    # --- Common properties ---
    @property    
    def theta_r(self) -> float:
        """Residual water content θr."""
        return self.props['theta_r']

    @property    
    def theta_s(self) -> float:
        """Saturated water content θs."""
        return self.props['theta_s']


    def sat_fr_theta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Saturation θ / n = θ / θ_s."""
        return self.theta / self.theta_s
    
    def sat_fr_S(self, S: float | np.ndarray)-> float |  np.ndarray:
        """Return saturation given S.
        
        Note that S = (theta - theta_r) / (theta_s - theta_r)
        and sat = theta / theta_s
        """
        return self.sat_fr_theta(self.theta_fr_S(S))
    
    # === Basic relations S_fr_theta and S_fr_psi ===
    # from defintion of S = (theta - theta_r) / (theta_S - theta_r)
    # S <--> theta
    def theta_fr_S(self, S: float | np.ndarray, eps: float = 1e-12)-> float | np.ndarray:
        """Return theta(S)"""        
        return self.theta_r + (self.theta_s - self.theta_r) * np.clip(S, eps, 1 - eps)

    def S_fr_theta(self, theta: float | np.ndarray, eps: float = 1e-12
                   )-> float | np.ndarray: 
        r"""Return $S(\theta)$"""                      
        return (np.clip(theta, self.theta_r + eps, self.theta_s - eps) - self.theta_r
                ) / (self.theta_s - self.theta_r)
        
    # === dS/dtheta dtheta/S (these are properties) ===
    def dS_dtheta(self, theta: float | np.ndarray)-> float | np.ndarray:
        """Return dS/dtheta
        
        dS/dtheta is a fixed number but we keep the size of theta to remain consistent
        with all other methods.
        """
        theta = np.atleast_1d(theta)
        dS_dth = np.ones_like(theta) / (self.theta_s - self.theta_r)
        return dS_dth if len(dS_dth > 1) else dS_dth.item()
        
    def dtheta_dS(self, S: float | np.ndarray)-> float | np.ndarray:
        """Return dtheta/dS
        
        dtheta/dS is a fixed number, but we keep the size of S to remain consistent
        with all other methods.
        """
        S = np.atleast_1d(S)
        dth_dS = np.ones_like(S) * (self.theta_s - self.theta_r)
        return dth_dS if len(dth_dS > 1) else dth_dS.item()
  
    def psispace(self, N: int = 50)-> np.ndarray:
        """Return a proper psi range for this soil (psi_b <= psi <= 10 ** 4.2)"""
        pF_max = 4.2
        psi_min = self.props['psi_b'] if 'psi_b' in self.props else 1.
        pF_min = np.log10(psi_min)
        return np.logspace(pF_min, pF_max, N)

    
    # === Basis Van Genughten and BC relation:  S <--> psi ===
    # The rest follows from these relations
    @abstractmethod
    def S_fr_psi(self, psi: float | np.ndarray) -> float | np.ndarray:
        """Return S(psi)"""
        pass

    @abstractmethod
    def psi_fr_S(self, S: float | np.ndarray) -> float | np.ndarray:
        """Return psi(S)"""
        pass
  
    # dS/dpsi,  dpsi/dS
    @abstractmethod
    def dS_dpsi(self, psi: float | np.ndarray)-> np.ndarray:
        """Return dS_dpsi"""
        pass
    
    @abstractmethod
    def dpsi_dS(self, S: float | np.ndarray)-> np.ndarray:
        """Return dpsi/dS"""        
        pass
    
    # dtheta_dpsi and dpsi_dtheta
    def dtheta_dpsi(self, psi: float | np.ndarray)-> np.ndarray:
        """Return dpsi/dS"""
        S = self.S_fr_psi(psi)
        return self.dtheta_dS(S) * self.dS_dpsi(psi)        

    def dpsi_dtheta(self, theta: float | np.ndarray)-> np.ndarray:
        """Return dpsi/dS"""
        S = self.S_fr_theta(theta)
        return self.dpsi_dS(S) * self.dS_dtheta(theta)
    
    # psi(theta) and theta(psi)
    def psi_fr_theta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Return psi_fr_theta."""
        S = self.S_fr_theta(theta)
        return self.psi_fr_S(S)        
        
    def theta_fr_psi(self, psi: float | np.ndarray) -> float:
        """Volumetric water content θ(h) as a function of suctioin head psi."""
        S = self.S_fr_psi(psi)
        return self.theta_fr_S(S)
        
  
    # K(theta) and K(psi)
    @abstractmethod
    def K_fr_S(S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
        pass
    
    def K_fr_theta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Hydraulic conductivity K(theta) as a function of theta."""
        S = self.S_fr_theta(theta)
        return self.K_fr_S(S)

    def K_fr_psi(self, psi: float | np.ndarray) -> float | np.ndarray:
        """Hydraulic conductivity K(psi) as a function of suction head psi."""
        S = self.S_fr_psi(psi)
        return self.K_fr_S(S)
        
    # ddK/dS, K/dtheta and dK/psi    
    @abstractmethod
    def dK_dS(S: float | np.ndarray)-> float | np.ndarray:
        """Return dK/dS"""
        pass
    
    def dK_dtheta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Return dK/dtheta""" 
        S = self.S_fr_theta(theta)
        return self.dK_dS(S) * self.dS_dtheta(theta)
        pass
    
    def dK_dpsi(self, psi: float | np.ndarray) -> float | np.ndarray:
        """Return dK/dpsi"""
        S = self.S_fr_psi(psi)
        return self.dK_dS(S) * self.dS_dpsi(psi)

    # available moister, theta_fc and theta_wp        
    def available_moisture(self, pfc: float = 2.5, pfw: float = 4.2) -> float:
        """Return available moisture to plants (between fc and wp)."""
        return self.theta_fc(pfc) - self.theta_wp(pfw)

    def theta_fc(self, pF: float =2.5)-> float:
        """Return theta at field capacity (where pF = 2.0 or 2.5). (vdMolen (1973))"""        
        psi_fc = 10 ** pF # cm
        return self.theta_fr_psi(psi_fc)

    def theta_wp(self, pF: float =2.5)-> float:
        """Return theta at wilting point (where pF = 4.2). (vdMolen (1973))"""        
        psi_wp = 10 ** pF # cm
        return self.theta_fr_psi(psi_wp)

    # C(psi) and C(theta), moisture capacity
    def C_fr_S(self, S: float | np.ndarray)-> float | np.ndarray:        
        return -self.dtheta_dpsi(self.psi_fr_S(S))
    
    def C_fr_psi(self, psi: float) -> float:
        """Specific moisture capacity C(psi) = dθ/dh = -dθ/dpsi."""
        return -self.dtheta_dpsi(psi)
        pass
    
    def C_fr_theta(self, theta: float) -> float:
        """Specific moisture capacity C(theta) = -dθ/dpsi."""    
        return -self.dtheta_dpsi(self.psi_fr_theta(theta))
    
    # D(psi) and D(theta): Diffusivity
    def D_fr_S(self, S: float | np.ndarray)-> float | np.ndarray:
        """Return diffusivity as a function of S"""
        return self.K_fr_S(S) / self.C_fr_S(S)
        
    def D_fr_psi(self, psi: float) -> float:
        """Diffusivity."""
        S = self.psi_fr_S(psi)        
        return self.D_fr_S(S)
    
    def D_fr_theta(self, theta: float) -> float:
        """Diffusivity.""" 
        S = self.theta_fr_S(theta)       
        return self.D_fr_S(S)

    # Throttling function that limit ET in rootmodel
    # ET reduction due to wet circumstances
    def alpha_wet(self, theta: float | np.ndarray,
                  psi1: float=10.0,
                  psi2: float=25.0) -> float | np.ndarray:
        """Return suffocation factor for wet conditions.
        
        Note that the input is theta, but the criteria are with psi1 and psi2.
        
        Parameters
        ----------
        theta: float or np.array
            moisture content
        psi1: float
            suction head below wich ET=0 (total suffocation), alpha=0
        psi2:
            suction head above which alpha=1 (no suffocation)
        """
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)
        alpha = np.zeros_like(psi)
        alpha[psi < psi1] = 0.0
        alpha[psi > psi2] = 1.0
        rng = (psi >= psi1) & (psi <= psi2)
        alpha[rng] = (psi[rng] - psi1) / (psi2 - psi1)
        if len(alpha) == 1:
            return alpha.item()
        else:
            return alpha

    # ET reduction according to Feddes
    def feddes_alpha(self, theta: float | np.ndarray,
                     psi1: float=10, psi2: float=25,
                     psi3:float =400, psi_w: float=16000) -> float | np.ndarray:
        """
        Compute Feddes reduction factor alpha(ψ) for given water content(s).
        Input is θ, converted internally to ψ using van Genuchten–Mualem relations.

        Feddes stress function (example parameters in cm suction head):
            psi < psi1        → 0.0 (saturation, anoxia)
            psi1 < psi < psi2 → linear increase to 1.0 (wet side)
            psi2 ≤ psi ≤ psi3 → 1.0 (optimal)
            psi3 < psi < psi_w→ linear decrease to 0.0 (dry side)
            psi ≥ psi_w       → 0.0 (permanent wilting)

        Parameters
        ----------
        theta : float or array_like
            Volumetric moisture content.
        psi1, psi2, psi3, psi_w : float
            Suction head thresholds (cm).

        Returns
        -------
        alpha : float or ndarray
            Stress reduction factor in [0, 1], same shape as input.
        """
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)  # convert θ → ψ
        alpha = np.zeros_like(psi, dtype=float)

        # Wet side: increase from 0 → 1
        mask_wet = (psi1 < psi) & (psi < psi2)
        alpha[mask_wet] = (psi[mask_wet] - psi1) / (psi2 - psi1)

        # Optimal zone: α = 1
        mask_opt = (psi2 <= psi) & (psi <= psi3)
        alpha[mask_opt] = 1.0

        # Dry side: decrease from 1 → 0
        mask_dry = (psi3 < psi) & (psi < psi_w)
        alpha[mask_dry] = (psi_w - psi[mask_dry]) / (psi_w - psi3)

        # Return scalar if input was scalar
        if alpha.size == 1:
            return alpha.item()
        return alpha

    # Smooth ET reduction, pratically same as Feddes
    def smooth_alpha(self, theta: float | np.ndarray,
                     s:    float = 0.2,
                     pF50: float = 3.6,
                     psi2: float = 25.
                     ) -> float | np.ndarray:
        """Return smooth ET throttling factor.

        Parameters
        ----------
        theta : float | array_like
            Volumetric moisture content.
        s : float, default=0.2
            Controls steepness of the dry-side logistic decline (0.15 < s < 0.30).
        pF50 : float, default=3.6
            pF at which alpha = 0.5 (typically 3.5 < pF50 < 3.8).
        psi2 : float, default=25
            Suction head [cm] at upper bound of wet side (Feddes psi2).

        Returns
        -------
        alpha : float or ndarray
            Reduction factor between 0 and 1.
        """
        theta = np.atleast_1d(theta)
        psi   = self.psi_fr_theta(theta)

        # pF definition
        pF  = np.log10(psi)
        pF2 = np.log10(psi2)
        pFw = 4.2   # wilting at pF ~ 4.2

        # Start with wet-side reduction (Feddes-style)
        alpha = self.alpha_wet(theta, psi2=psi2)

        # Set to zero beyond wilting
        alpha[pF > pFw] = 0.0

        # Logistic decline on dry side
        rng = (pF >= pF2) & (pF < pFw)
        alpha[rng] = 1.0 / (1.0 + np.exp((pF[rng] - pF50) / s))

        # Return scalar if input was scalar
        return alpha.item() if alpha.size == 1 else alpha
            
    # ET reduction according to Mualem
    def mualem_alpha(self, theta: float | np.ndarray,
                     beta: float =1.0,
                     psi2: float =25,
                     psi_fc: float=300,
                     psi_w: float=16000) -> float | np.ndarray:
        """Return smooth ET throttling factor acc Mualem.

        Parameters
        ----------
        theta: float, ndarray    
            moisture content        
        beta: tunes curvature
            0.7 < beta < 1.3, beta > 1 steeper, beta < 1 gentler
        """        
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)
        
        alpha = np.ones_like(psi)
                
        K_fc = max(self.K_fr_theta(self.theta_fr_psi(psi_fc)), 1e-30)

        rng = (psi > psi_fc) & (psi < psi_w)
        alpha[rng] = (self.K_fr_theta(self.theta_fr_psi(psi[rng])) / K_fc) ** beta
        alpha[psi >= psi_w] = 0

        alpha  *= self.alpha_wet(theta, psi2=psi2)

        # Return scalar if input was scalar
        return alpha.item() if alpha.size == 1 else alpha

    # ET reducution, blend of Feddes and Mualem
    def alpha_dry_blend(self, theta: float | np.ndarray,
                        beta: float=1.0,
                        w: float =0.6,
                        psi1: float=10,
                        psi2: float=25,
                        psi3: float=400,
                        psi_w: float=16000,
                        eps: float=1e-12) -> float | np.ndarray:
        """
        Dry-side throttle: geometric blend of Mualem and Feddes.
        w=1 → pure Mualem, w=0 → pure Feddes.
        """
        psi_fc, psi_w = 300, 16000
        theta = np.atleast_1d(theta)
        
        # masks
        # dry = (theta > th_wp) & (theta < th_fc)

        a_mualem = self.mualem_alpha(theta, beta=beta, psi2=psi2, psi_fc=psi_fc, psi_w=psi_w)        
        a_feddes = self.feddes_alpha(theta, psi1=psi1, psi2=psi2, psi3=psi3, psi_w=psi_w)
        a_m = np.clip(a_mualem, eps, 1.0)
        a_f = np.clip(a_feddes, eps, 1.0)

        alpha = np.exp(w * np.log(a_m) +  (1 - w) * np.log(a_f))
        
        # enforce bounds
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha.item() if alpha.size == 1 else alpha
    
    @cached_property
    def S_fr_lnK(self)-> None:
        """Return intepolator to get k at given S
        pass
        
        must be in __init__ as it generateds an Interpolator
        to get the S for a given K-value
        
        """
        S_limit = 1e-4      
        S = np.linspace(S_limit, 1, 20)
        # Dummy k
        K = self.K_fr_S(S)
        
        # Remove compuational artefacts
        S = S[K > 0]
        K = K[K > 0]
        
        # Guarantee that k always rises
        rises = np.ones_like(K, dtype=bool)
        kmax = 0
        for i, k in enumerate(K):
            if k > kmax:
                kmax = k
            else:
                rises[i] = False
        K = K[rises]
        S = S[rises]
        return PchipInterpolator(np.log(K), S)        

    def S_fr_K(self, k: float | np.ndarray)-> float | np.ndarray:
        """Return S(K), saturation given K (by interpolation) """
        return self.S_fr_lnK(np.log(k))
    
    @cached_property
    def theta_fr_lnV(self)-> None:
        """Return theta(v), where V = dK(theta)_dtheta.
        
        To compute the theta in the Kinematic Wave approach for given
        velocity v = dk(theta)/dtheta we generate an interpolator.
        """        
        theta = np.linspace(self.theta_r, self.theta_s)
        V = self.dK_dtheta(theta)
        
        # Remove compuatational artefacts
        theta  = theta[V > 0]
        V      = V[V > 0]
        
        #Guarantee that V always rises
        rises = np.ones_like(V, dtype=bool)
        vmax = 0.        
        for i, v in enumerate(V):
            if v > vmax:
                vmax = v
            else:
                rises[i] = False
        V     = V[rises]
        theta = theta[rises]
        return PchipInterpolator(np.log(V), theta)
    
    def theta_fr_V(self, v: float | np.ndarray)-> float | np.ndarray:
        """Return theta(V), where V = dK(theta)_dtheta"""
        if self.theta_fr_lnV is None:
            self.set_theta_fr_lnV_interpolator()            
        return self.theta_fr_lnV(np.log(v))

    
    def theta_fr_K(self, k: float | np.ndarray)-> float| np.ndarray:
        """Return theta(K), theta given K."""
        return self.theta_fr_S(self.S_fr_K(k))

    def get_vD(self, q_avg: float)-> tuple[float, float]:
        """Return v = dK_dtheta(theta_avg)) and D(theta_abg) =k0(theta_avg) h1(theta_avg)"""        
        k0 = q_avg
        S = self.S_fr_K(k0).item()        
        v = self.dK_dtheta(self.theta_fr_S(S))
        D = self.D_fr_S(S)
        return v.item() if len(v) == 1 else v, D.item() if len(D) == 1 else D
    
    
    def IR(self, z: float | np.ndarray, tau: np.ndarray, q_avg: float =None)-> float | np.ndarray:
        """Return impulse response of q_avg with the linearized convection dispersion equation for flux.

        $$ \frac{partial q}{\\partial \tau} = -v \frac{\\partial q}{\\partial z} + D \frac {\\partial^2 q}{\\partial z^2}$$

        Parameters
        ----------
        z: ndarray
        depth below the root zone
        tau: ndarray
        time
        v: float [L/T]
        vertical front velocity (first derivative of conductivity at $\theta = \theta_{avg}$)
        D: diffusivity, float [L^2/T]
        equals k0 h1, where k0 is conductivity at theta_avg and h1 is the first
        derivative ($dh/theta$) at $theta=theta_{avg}$
        q_avg: average recharge (passage through the percolation zone) [L/T]
        """
        v, D = self.get_vD(q_avg)  
        arg = np.clip((z -  v * tau) ** 2 / (2 * D * tau), -50, 50)
        return z / np.sqrt(4 * np.pi * D * tau ** 3) * np.exp(-arg)


    def SR_Phi(self, z: float | np.ndarray, tau: np.ndarray, q_avg: float, clip: float =50)-> float | np.ndarray:
        """Return step response of q_avg with the linearized convection dispersion equation for flux.

        The step response is here computed using the exact soution with
        the cumulative normal distribution.
        """    
        v, D = self.get_vD(q_avg)
        arg1 = np.clip( (z - v * tau) / np.sqrt(2 * D * tau), -clip, clip)
        arg2 = np.clip(-(z + v * tau) / np.sqrt(2 * D * tau), -clip, clip)
        phi1 = norm.cdf(arg1)
        phi2 = norm.cdf(arg2)
        factor = np.exp(np.clip(v * z / D, -clip, +clip))
        return 1 - (phi1 - factor * phi2)

    def SR_erfc(self, z: float | np.ndarray, tau: np.ndarray, q_avg: float, clip: float=50)-> float | np.ndarray:
        """Return step response of q with the linearized convection dispersion equation for flux.
        Maas (1994) eq. 3.25

        The step response is here computed using the exact solution with erfc's.
        """    
        v, D = self.get_vD(q_avg)
        sqrt4Dt = np.sqrt(4.0 * D * tau)
        A = np.clip((z - v * tau) / sqrt4Dt, -clip, clip)
        B = np.clip((z + v * tau) / sqrt4Dt, -clip, clip)
        return 0.5 * (erfc(A)  + np.exp(np.clip(v * z / D, -clip, clip)) * erfc(B))

    def IR_PIII(self, z: float | np.ndarray, tau: np.ndarray, q_avg: float, clip: float=50)-> float | np.ndarray:
        """Return Peason III distribution based on the moments.

        Parameters
        ----------
        z: ndarray or float
        depth below the root zone.
        tau: ndarray or float
        time.
        q_avg: float
        mean percolation rate.
        """
        v, D = self.get_vD(q_avg)

        M1 = z / v                   # first moment
        M2 = 2 * z * D / v ** 3      # second central moment
        M3 = 12 * z * D ** 2 / v ** 5 # third central moment

        a = 2 * M2 / M3
        b = M1 - 2 * M2 ** 2 / M3
        n = 4 * M2 ** 3 / M3 ** 2

        # check
        # aa = v ** 2 / (3 * D)
        # bb = - z / (3 * v)
        # nn = 2 * z * v / (9 * D)    
        # print(a, aa)
        # print(b, bb)
        # print(n, nn)

        pIII = np.zeros_like(tau)
        tau_r = tau[tau >= b] - b
        
        # Use log to prevent overflow of Gamma(n) and (t-b)^n
        log_p = (n * np.log(a) + (n - 1) * np.log(tau_r) -
                                    a * tau_r  - gammaln(n))
        
        pIII[tau >= b] = np.exp(log_p)
        
        return pIII


    def SR_PIII(self, z: float | np.ndarray, tau: np.ndarray, q_avg: float)-> float | np.ndarray:
        """Pierson III als step response

        The integration of IR_PIII_BC is known and is astandard function,
        which is used here.
        """
        v, D = self.get_vD(q_avg)

        z = -z

        M1 = - z / v                   # first moment
        M2 = - 2 * z * D / v ** 3      # second central moment
        M3 = -12 * z * D ** 2 / v ** 5 # third central moment

        a = 2 * M2 / M3
        b = M1 - 2 * M2 ** 2 / M3
        n = 4 * M2 ** 3 / M3 ** 2

        F = np.zeros_like(tau)
        F[tau >= b] = gamma.cdf(tau[tau >= b] - b, a=n, scale=1/a)
        return F

    @staticmethod
    def BR(sr_func: Callable[[float | np.ndarray, np.ndarray], float | np.ndarray],  z: float | np.ndarray, tau: float | np.ndarray, q_avg: float)-> np.ndarray:
        """Return the block response using step response function sr_func.

        Parameters
        ----------
        sr_func: function
        step response function (must take z, tau, q_avg)
        z: float [m] or ndarray
        depth
        tau: ndarray (starting with 0) [d]
        time
        q_avg: float [m/d]
        average recharge
        """
        BR = np.hstack((0., sr_func(z, tau[1:], q_avg)))
        BR[1:] -= BR[:-1]
        return BR


