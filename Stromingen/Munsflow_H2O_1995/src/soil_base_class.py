# Soil base Class

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class SoilBase(ABC):
    """Abstract base class for soil hydraulic properties."""

    def __init__(self, soil_code: str) -> None:
        if SoilBase.data is None:
            raise RuntimeError("Load soils first with Soil.load(filename)")
        try:
            self.props = SoilBase.data.loc[soil_code].copy()
        except (ValueError, IndexError):
            raise IndexError(f"Soil code {soil_code} unknown, not in soils")
        
    # --- Required methods for subclasses ---
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
    
    @abstractmethod
    def S_fr_theta(self, theta: float | np.ndarray,  eps: float =1e-12) -> float | np.ndarray:
        """Return S(theta)"""
        pass
    
    @abstractmethod
    def S_fr_psi(self, psi: float | np.ndarray) -> float | np.ndarray:
        """Return S(psi)."""
        pass
    
    @abstractmethod
    def psi_fr_theta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Return psi_fr_theta."""
        pass
    
    @abstractmethod
    def theta_fr_psi(self, h: float | np.array) -> float:
        """Volumetric water content θ(h) as a function of suctioin head psi."""
        pass

    @abstractmethod
    def K_fr_theta(self, theta: float | np.ndarray) -> float | np.ndarray:
        """Hydraulic conductivity K(theta) as a function of theta."""
        pass

    @abstractmethod
    def K_fr_psi(self, psi: float | np.ndarray) -> float | np.ndarray:
        """Hydraulic conductivity K(psi) as a function of suction head psi."""
        pass

    @abstractmethod
    def dK_dtheta(self, theta: float | np.ndarray) -> float | np.ndarray:
        pass

    @abstractmethod
    def dpsi_dtheta(self, theta: float | np.ndarray) -> float| np.ndarray:
        pass

    @abstractmethod
    def dtheta_dpsi(self, psi: float | np.ndarray) -> float| np.ndarray:
        pass
    
    @abstractmethod
    def theta_fc(self, pF: float = 2.5) -> float:
        """Water concent at field capacity, default pF=2.5"""
        pass
    
    @abstractmethod
    def theta_wp(self, pF: float = 4.2) -> float:
        """Water content at wilting point, default pF=4.2"""
        pass
    
    @property
    @abstractmethod
    def available_moisture(self) -> float:
        pass
    
    @abstractmethod
    def C_fr_theta(self, theta: float) -> float:
        """Specific moisture capacity C(theta) = -dθ/dpsi."""
        pass

    @abstractmethod
    def C_fr_psi(self, psi: float) -> float:
        """Specific moisture capacity C(psi) = -dθ/dpsi."""
        pass
    
    @abstractmethod
    def D_fr_theta(self, theta: float) -> float:
        """Diffusivity."""        
        return self.K_fr_theta(theta) / self.C_fr_theta(theta)
    
    @abstractmethod
    def D_fr_psi(self, psi: float) -> float:
        """Diffusivity."""        
        return self.K_fr_psi(psi) / self.C_fr_psi(psi)

    
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
            return np.item(alpha)
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
        return np.item(alpha) if alpha.size == 1 else alpha
            
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
                
        K_fc = max(self.K(self.theta_fr_psi(psi_fc)), 1e-30)

        rng = (psi > psi_fc) & (psi < psi_w)
        alpha[rng] = (self.K(self.theta_fr_psi(psi[rng])) / K_fc) ** beta
        alpha[psi >= psi_w] = 0

        alpha  *= self.alpha_wet(theta, psi2=psi2)

        # Return scalar if input was scalar
        return np.item(alpha) if alpha.size == 1 else alpha

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


    # --- Common properties ---
    @property    
    def theta_r(self) -> float:
        """Residual water content θr."""
        return self.props['theta_r']

    @property    
    def theta_s(self) -> float:
        """Saturated water content θs."""
        return self.props['theta_s']

    @property    
    def saturation(self) -> float:
        """Saturation degree (θs - θr)."""
        return self.theta_s - self.theta_r












import math

class VanGenuchtenSoil(SoilBase):
    def __init__(self, theta_r, theta_s, alpha, n, Ks):
        super().__init__(theta_r, theta_s)
        self.alpha = alpha
        self.n = n
        self.m = 1 - 1/n
        self.Ks = Ks

    def Se(self, h: float) -> float:
        """Effective saturation."""
        if h >= 0:
            return 1.0
        return (1 + (self.alpha * abs(h))**self.n) ** -self.m

    def theta(self, h: float) -> float:
        return self.theta_r + (self.theta_s - self.theta_r) * self.Se(h)

    def K(self, h: float) -> float:
        Se = self.Se(h)
        return self.Ks * (Se ** 0.5) * (1 - (1 - Se ** (1/self.m))**self.m) ** 2

    def C(self, h: float) -> float:
        """Derivative of θ(h) wrt h."""
        if h >= 0:
            return 0.0
        term = (self.alpha * abs(h))**self.n
        Se = (1 + term)**-self.m
        dSedh = -self.alpha * self.n * self.m * (self.alpha*abs(h))**(self.n-1) \
                * Se**(1/self.m + 1) * (1/abs(h))
        return (self.theta_s - self.theta_r) * dSedh
