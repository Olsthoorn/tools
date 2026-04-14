"""
# Aanleiding
Dit bestand bevat de implementatie van de functies die gebruikt worden voor de analytische berekening
van grondwaterstandsverlagingen ten behoeven van de update van de Voortoets.

Deze functionaliteit wordt gebruikt in het notebook VoorToetsIllustraties.iynb.

@T.N.Olsthoorn Maart 2025
"""
# %%
import os
from abc import ABC, abstractmethod
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from scipy.integrate import quad
from scipy.signal import filtfilt
from scipy.special import erfc, exp1
from scipy.special import k0 as K0
from scipy.special import k1 as K1

# %% --- Basic functionality
class Dirs:
    """Namespace for directories in project"""
    def __init__(self):
        self.home = '/Users/Theo/Entiteiten/Hygea/2022-AGT/jupyter/'
        self.data = os.path.join(self.home, 'data')
        self.images = os.path.join(self.home, 'images')
    
dirs = Dirs()
os.path.isdir(dirs.data)
os.path.isdir(dirs.images)


def newfig(title, xlabel, ylabel, xlim=None, ylim=None, xscale=None, yscale=None, figsize=None):
    """Set up a new figure with a single axes and return the axes."""
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if xscale is not None:
        ax.set_xscale(xscale)
    if yscale is not None:
        ax.set_yscale(yscale)
    ax.grid(True)
    return ax

# --- Theis and Hantush well functions
def Wt(u):
    """Return Theis well function.
    
    Wh(u, 0) also yields Wt(u)
    
    Parameters
    ----------
    u: float, np.ndarray
        u = r^2 S / (4 kD t)
    """
    return exp1(u)

def Wh(u, rho=0):
    """Return Hantush's well function values.
    
    Parameters
    ----------
    u = r^s / (4 kD t)
    rho = r / lambda,    lambda = sqrt(kD c)
    
    >>>Wh(0.004, 0.03)
    array(4.8941042)
    
    >>>Wh(0.05625, 0.03)
    array(2.35295487)

    """
    def integrant(y, rho):
        """Return the function to be integrated."""
        return np.exp(-y - (rho / 2) ** 2 / y) / y
    
    def w(u, rho): # Integrate the argument
        return quad(integrant, u, np.inf, args=(rho,))[0]
    
    wh = np.vectorize(w) # Vectorize function w(u, rho) so we can use arrays as input.
    
    return np.asarray(wh(u, rho), dtype=float)


def Wb(tau, rho=0):
    """Return Hantush well function values using the Bruggeman (1999) form of Hantush's Well function.
    
    Bruggeman (1999) prefers , using a somewhat different form of Hantush's Well function, one in which
    time and distance to the well are truly separated. However, this loses a bit the connection with
    Theis, which we have always been used to. Bruggeman uses W(tau=t / (cS), rho=r / lambda)
    instead of W(u=r^2 S / (4 kD t), rho=r / lambda)
    
    Parameters
    ----------
    tau = t/ (cS)
    rho = r / lambda, lambda= sqrt(kD c)
    
    >>>u, rho = 0.004, 0.03
    >>>tau = rho ** 2 /(4 * u)
    >>>Wb(tau, rho)
    array(4.8941042)  
    """
    def integrant(x, rho):
        """Return the function to be integrated."""
        return np.exp(-x - (rho / 2) ** 2 / x) / x
    
    def w(tau, rho):
        """Integrate the argument."""
        return quad(integrant, 0, tau, args=(rho,))[0]
    
    # Vectorize function w(u, rho) so we can use arrays as input.
    wh = np.vectorize(w)
    return np.asarray(wh(tau, rho), dtype=float)


def Wb1(tau, rho=0):
    """Return Hantush well function values using the Bruggeman (1999) form of Hantush's Well function.
    
    This function just converts the input paramters of Bruggean into
    those of the regular Hantush well function. It's the simplest conversion,
    no extra functionality is needed this way.
    
    Parameters
    ----------
    tau = t/ (cS)
    rho = r / lambda, lambda= sqrt(kD c)
    
    >>>u, rho = 0.004, 0.03
    >>>tau = rho ** 2 /(4 * u)
    >>>Wb1(tau, rho)
    array(4.8941042)  

    >>>Wb1(0.05625, 0.03)
    array(4.8941042)  
    """
    u = rho ** 2  / (4 * tau)
    return Wh(u, rho)


def SRtheis(T=None, S=None, r=None, t=None):
    """Return Step Response for the Theis well function.
    
    The step response is a function of time (tau).
    It's just the Theis drawdown for unit extraction (Q=0).
    Result = 0 for tau=0 is guaranteed.
    """
    dt = np.diff(t)
    assert(np.all(np.isclose(dt[0], dt))), "all dt must be the same."
    u = r ** 2 * S / (4  * T * t[1:])
    return np.hstack((0, 1 / (4 * np.pi * T) * exp1(u)))


def BRtheis(T=None, S=None, r=None, t=None):
    """Return Block Response for the Theis well function.
    
    The block response is a function of time (tau).
    Results is 0 for tau = 0 is guaranteed.
    """
    SR = SRtheis(T, S, r, t)
    return np.hstack((0, SR[1:] - SR[:-1]))


def IRtheis(T=None, S=None, r=None, t=None):
    """Return the impulse response for the Theis well drawdown.
    
    The impulse response is the derivative of the step response.
    In practice it only has theoretical value.
    It can be used instead of the block response, but then the
    time step must be sufficiently small, much smaller than the
    usual time step of one day.
    """
    dt = np.diff(t)
    assert np.all(np.isclose(dt[0], dt))
    u = np.hstack((np.nan, r ** 2 * S / (4 * T * t[1:])))
    return np.hstack((0, 1 / (4 * np.pi * T) *  np.exp(-u[1:]) / t[1:])) * dt[0]

# Check Hantush and Bruggeman functions for a well in a leaky aquifer.
# u, rho = 0.004, 0.03
# print("Wh(u={:.4g}, rho={:.4g}) = {}".format(u, rho, Wh(u, rho)))
# tau = rho ** 2 / (4 * u)
# print("Wb(tau={:.4g}, rho={:.4g}) = {}".format(tau, rho, Wb(tau, rho)))


# --- Wells
class WellBase(ABC):
    """Base class for all well functions that follow.
    
    The abstractmethod decorator makes sure that all
    well subclasses implement these method.
    """
        
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        """Initialize a well.
        
        Parameters
        ----------
        xw, yw: float
            Well location coordinates.
        rw: float, Optional
            well radius
        z1, z2: float, optional
            Depth boundaries of the well.
        aqprops: dict, optional
            Aquifer propertiers (see below).
            
        Returns
        -------
        a member of the Well class
        """
        if not (np.isscalar(xw) and np.isscalar(yw)):
            raise ValueError("xw and yw must both be scalars.")
        self.xw = xw
        self.yw = yw
        self.rw = rw
        self.z1 = z1
        self.z2 = z2
        self.aq = aqprops
        return
    
    @abstractmethod
    def dd(self, *args, **kwargs):
        """Return drawdown for well."""
        pass
    
    @abstractmethod
    def h(self, *args, **kwargs):
        """Return head in water table aquifer."""
        pass
    
    @abstractmethod
    def Qr(self, *args, **kwargs):
        """Return the discharge across a circle with radius r."""
        pass
    
    @abstractmethod
    def qxqy(self, *args, **kwargs):
        """Return specific discharge vectors qx and qy."""
        pass

    
    @staticmethod    
    def itimize(a):
        """Return a as a scalar if a is an array and len(a) == 1"""
        if isinstance(a, np.ndarray) and len(a) == 1:
            a = a.item()
        return a
    
    @staticmethod
    def check_pars_presence(dict):
        none_item_keys=set()
        for k, item in dict.items():
            if item is None:
                none_item_keys.add(k)
        if len(none_item_keys) > 0:            
                raise ValueError(f"{none_item_keys} must not be None!")
        return

    @staticmethod
    def check_xyt(x, y, t):
        """Return check of x, y and t.
        
        x and y must both be floats of t is an array and vice versa.
        If x and y are arrays they must have the same shape.
        y may be None and in which case it is ignored,
        """
        if isinstance(t, np.ndarray):
            if not np.isscalar(x):
                raise ValueError("x must be scalar if t is an array!")
            if y is not None:
                if not np.isscalar(y):
                    raise ValueError("y must be a scalar if x is a scalar!")
        else:
            x, y = WellBase.check_xy(x, y)
        return x, y, t
    
    @staticmethod
    def check_xy(x, y):
        """Return checked coordinates x and y
        
        x and y are turned into arrays.
        The must have the same shape.
        y will be ignored if None
        """
        x = np.atleast_1d(x).astype(float)
        if y is not None:
            y = np.atleast_1d(y).astype(float)
            if not np.all(x.shape == y.shape):
                raise ValueError("x.shape must equal y.shape!")
        return x, y
    
    @staticmethod
    def check_keys(subset, aqprops):
        """Check and update the aquifer properties according to subset.
        
        Subset is the set of required keys. It is checked if these are
        in the dict aqprops. If not it is tried to update aqprops. If
        that is not possible a ValueError is raised.
        """
        if {'k', 'D'}.issubset(aqprops):
            aqprops['kD'] = aqprops['k'] * aqprops['D']
        if 'kD' not in aqprops.keys():
            raise ValueError("Neither kD, k nor D in aqprops!")

        if 'S' in subset and 'S' not in aqprops.keys():
            raise ValueError("S not in aqprops!")
        
        # kD is guaranteed so make c and lambda consistent if either c or lambda in aqprops
        if 'c' in aqprops.keys():
            aqprops['lambda'] = np.sqrt(aqprops['kD'] * aqprops['c'])
        elif 'lambda' in aqprops.keys():
            aqprops['c'] = aqprops['kD'] ** 2  / aqprops['lambda']
                              
        # Then if c or lambda is required by subset but still not in aqprops --> ValueError  
        if 'c' in subset:
            if 'c' not in aqprops.keys():
                    raise ValueError("Neither c nor lambda in aqprops")
            
        if 'lambda' in subset:
            if 'lambda' not in aqprops.keys():
                    raise ValueError("Neither c nor lambda in aqprops!")
                
    
    def radius(self, x=None, y=None):
        """Return the distance between well and x, and y.
        
        Parameters
        ----------
            x : np.ndarray
                x-coordinates
            y: np.ndarray like x or None
                y-coordinates or None, when only x is used
        """
        x = np.atleast_1d(x).astype(float)
        if y is None:
            r = np.sqrt((self.xw - x) ** 2)
        else:
            y = np.atleast_1d(y).astype(float)
            if not np.all(x.shape == y.shape):
                raise ValueError("x and y must have same shape")
            try:                
                r = np.sqrt((self.xw - x) ** 2 + (self.yw - y) ** 2)
            except Exception as e:
                print(e)
                raise
        r[r < self.rw] = self.rw
        return r
    
class wTheis(WellBase):
    """Class for handling drawdown and other calcuations according to Theis"""
    
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD, S'}, self.aq)
        return
    
    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for Theis well type.")

    def dd(self, x=None, y=None, t=None, Q=None):
        """Return well's drawdown according to Theis.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.
        
        Parameters
        ----------
        Q: float
            constant extraction by the well
        x: float or np.ndarray of floats
            x coordinate[s] of drawdown point.
        y: None or float or np.ndarray
            y coordinate[s] or None
        t: float or np.ndarray
            times to compute dd (all dd[t <= 0] =0)        
        """
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        
        FQ = Q / (4 * np.pi * self.aq['kD'])
        
        if isinstance(t, np.ndarray):
            u = np.zeros_like(t)
            s = np.zeros_like(u)
            u = r ** 2 * self.aq['S']  / (4 * self.aq['kD'] * t[t > 0])
            s[t > 0] = FQ * Wt(u)            
        else:
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)
            s = FQ * Wt(u)
        
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None, y=None, t=None):
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        if isinstance(t, np.ndarray):
            Qr_ = np.zeros_like(t) + Q
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t[t > 0])
            Qr_[t > 0] = Q * np.exp(-u)
        else:
            Qr_ = np.zeros_like(r) + Q
            u = r[r > 0] ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)
            Qr_[r > 0] = Q * np.exp(-u)
            
        Qr_ = WellBase.itimize(Qr_)        
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None, t=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y, t)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
    
    def SR(self, r=None, t=None):
        """Return Step Response for the Theis well function.

        The step response is a function of time (tau).
        It's just the Theis drawdown for unit extraction (Q=0).
        Result = 0 for tau=0 is guaranteed.
        """
        dt = np.diff(t)
        if not np.all(np.isclose(dt[0], dt)):
            raise ValueError("Stepsize must all be the same!")
        u = r ** 2 * self.aq['S'] / (4  * self.aq['kD'] * t[1:])
        return np.hstack((0, 1 / (4 * np.pi * self.aq['kD']) * exp1(u)))

    def BR(self, r=None, t=None):
        """Return Block Response for the Theis well function.

        The block response is a function of time (tau).
        Results is 0 for tau = 0 is guaranteed.
        """
        sr = self.SR(r=r, t=t)
        return np.hstack((0, sr[1:] - sr[:-1]))

class wTheisSimple(WellBase):
    """Class for handling drawdown and other calcuations according to simplified Theis Function
    
    The simplified Theis well fuction is ln (2.25 kD t/ (r ^2 S))
    Or with gamma=0.577216, s = ln(e^(-0.577216) / u) = 'ln(0.5615  / u)
    """
    
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD, S'}, self.aq)
        self.gamma = 0.577216
        return
    
    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for Theis well type.")

    def dd(self, x=None, y=None, t=None, Q=None):
        """Return well's drawdown according to Theis.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.
        
        Parameters
        ----------
        Q: float
            constant extraction by the well
        x: float or np.ndarray of floats
            x coordinate[s] of drawdown point.
        y: None or float or np.ndarray
            y coordinate[s] or None
        t: float or np.ndarray
            times to compute dd (all dd[t <= 0] =0)        
        """        
        
        def W(u):
            """Return simplified Theis well function."""
            u = np.atleast_1d(u).astype(float)            
            gu = np.exp(-self.gamma) /u
            w = np.zeros_like(u)
            w[ gu >= 1] = np.log(gu)
            return w
        
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        
        FQ = Q / (4 * np.pi * self.aq['kD'])
        
        if isinstance(t, np.ndarray):
            u = np.zeros_like(t)
            s = np.zeros_like(u)
            u = r ** 2 * self.aq['S']  / (4 * self.aq['kD'] * t[u > 1])
            s[t > 0] = FQ * W(u)
        else:
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)    
            s = FQ * W(u)
        
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None, y=None, t=None):
        x, y, t = WellBase.check_xyt(x, y, t)
        
        def fq(u):
            """Return approximate flow for Q=1.
            
            Notice that for the simplified Theis solution,
            Q=Q for gu > 1 and 0 for gu <= 1.            
            """
            u = np.atleast_1d(u).astype(float)
            gu = np.exp(-self.gamma) / u
            q = np.zeros_like(u)
            q [gu > 1] = 1.
            return q

        r = self.radius(x, y)
        if isinstance(t, np.ndarray):
            Qr_ = np.zeros_like(t) + Q
            u = r ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t[t > 0])
            Qr_[t > 0] = Q * fq(u)
        else:
            Qr_ = np.zeros_like(r) + Q
            u = r[r > 0] ** 2 * self.aq['S'] / (4 * self.aq['kD'] * t)
            Qr_[r > 0] = Q * fq(u)
            
        Qr_ = WellBase.itimize(Qr_)        
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None, t=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y, t)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
    
    def SR(self, r=None, t=None):
        """Return Step Response for the simplified Theis well function.

        The step response is a function of time (tau).
        It's just the Theis drawdown for unit extraction (Q=0).
        Result = 0 for tau=0 is guaranteed.
        """
        dt = np.diff(t)
        if not np.all(np.isclose(dt[0], dt)):
            raise ValueError("Stepsize must all be the same!")
        u = r ** 2 * self.aq['S'] / (4  * self.aq['kD'] * t[1:])
        gu = np.exp(-self.gamma) / u
        gu[gu < 1] = 1.
        sr = np.hstack((0, 1 / (4 * np.pi * self.aq['kD']) * np.log(gu)))
        return WellBase.itimize(sr[sr > 0.]) # Truncated where sr=0.

    def BR(self, r=None, t=None):
        """Return Block Response for the Theis well function.

        The block response is a function of time (tau).
        Results is 0 for tau = 0 is guaranteed.
        """
        sr = self.SR(r=r, t=t)
        return np.hstack((0, sr[1:] - sr[:-1]))

class wHantush(WellBase):
    """Clas for computing drawdown and other values according to Hantush."""
    
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD', 'c'}, self.aq)

    def h(self, Q=None, x=None, y=None, t=None):
        """Return head in wate table aquifer."""
        raise NotImplementedError("h not implemented for Hantush well type.")


    def dd(self, x=None, y=None, t=None, Q=None):
        """Return well's drawdown according to Theis.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.
        
        We use Bruggemans function Wb(tau, rho) because this separates
        time and disrance with u and rho not.
        
        Parameters
        ----------
        Q: float
            constant extraction by the well
        x: float or np.ndarray of floats
            x coordinate[s] of drawdown point.
        y: None or float or np.ndarray
            y coordinate[s] or None
        t: float or np.ndarray
            times to compute dd (all dd[t <= 0] =0)        
        """
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
                            
        FQ = Q / (4 * np.pi * self.aq['kD'])
        
        if isinstance(t, np.ndarray):
            s = np.zeros_like(t)
            tau = t[t > 0] / (self.aq['c'] * self.aq['S'])
            rho = r  / self.aq['lambda']
            s[t > 0] = FQ * Wb(tau, rho)            
        else:
            tau = t / (self.aq['c'] * self.aq['S'])
            rho = r  / self.aq['lambda']
            s = FQ * Wb(tau, rho)
        
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, Q=None, x=None, y=None, t=None, dr=1e-3):
        """Return the Qr for the Hantush well.
        
        Note, if t is an array, x (and y of not None) must be scalars.
        If t is a scalar, x (and y if not None) may be an array.

        We do this numerically because there is no simple analytical solution for it.        
        """        
        x, y, t = WellBase.check_xyt(x, y, t)
        
        r = self.radius(x, y)
        
        tau = t / (self.aq['c'] * self.aq['S'])                                                
        if isinstance(t, np.ndarray):
            Qr_ = np.zeros_like(t)
            Qr_[tau > 0] = Q * (Wb(tau[t > 0], (r + dr) / self.aq['lambda']) -
                       Wb(tau[t > 0], (r - dr) / self.aq['lambda'])) * r  / dr
        else:
            Qr_ = np.zeros_like(r)            
            Qr_[r > 0] = Q * (Wb(tau, (r[r > 0] + dr) / self.aq['lambda'] -
                        Wb(tau, (r[r > 0] - dr) / self.aq['lambda']))) * r[r > 0]  / dr
        
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, Q=None, x=None,  y=None, t=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xyt(x, y, t)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
    
    def SR(self, r=None, t=None):
        """Return Step Response for the Hantus well function.

        The step response is a function of time (tau).
        It's just the Theis drawdown for unit extraction (Q=0).
        Result = 0 for tau=0 is guaranteed.
        """
        dt = np.diff(t)
        if not np.all(np.isclose(dt[0], dt)):
            raise ValueError("Stepsize must all be the same!")
        u = r ** 2 * self.aq['S'] / (4  * self.aq['kD'] * t[1:])
        rho = r / self.aq['lambda']
        return np.hstack((0, 1 / (4 * np.pi * self.aq['kD']) * Wh(u, rho)))

    def BR(self, r=None, t=None):
        """Return Block Response for the Theis well function.

        The block response is a function of time (tau).
        Results is 0 for tau = 0 is guaranteed.
        """
        sr = self.SR(r=r, t=t)
        return np.hstack((0, sr[1:] - sr[:-1]))

class wDupuit(WellBase):
    """General implementation of the Dupuit well function."""
    
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        """Return drawdown according to Dupuit.
        
        The drawdown in a confined or unconfined aquifer with fixed head
        or zero drawdown at r=R without recharge.

        Notice that this the same as Verruijt with N=0.
        """
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD'}, self.aq)
        return

    def h(self, Q=None, x=None, y=None, R=None):
        """Return wetted aquifer thickness according to Dupuit.
        
        This is the same as Verruijt for N=0.        
        """
        WellBase.check_pars_presence({'R': R})
        WellBase.check_keys({'k', 'D'}, self.aq)
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)

        H = self.aq['D']
        
        FQ = Q / (np.pi * self.aq['k'])
        
        h2 = np.zeros_like(r) + H ** 2       
        h2[r <= R] = H ** 2  - FQ * np.log(R / r[r <= R])
        h2[h2 < 0] = 0.
        return WellBase.itimize(np.sqrt(h2))        


    def dd(self, x=None, y=None, Q=None, R=None):
        """Return drawdown according to Dupuit.
        
        This is the same as Verrijt for N=0.        
        """
        WellBase.check_pars_presence({'R': R})
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s = np.zeros_like(r)
        s[r <= R] = FQ * np.log(R / r[r <= R])
        return WellBase.itimize(s)
    
    def Qr(self, Q=None, x=None, y=None, R=None):
        """Return the flow at a distance given by x and y.
        
        This is the same as Verrijt for N=0 --> just Q.
        """
        WellBase.check_pars_presence({'R': R})
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        
        Qr_ = np.zeros_like(r)
        Qr_[r <= R] = Q
        
        return WellBase.itemize(Qr_)
    
    def qxqy(self, Q=None, x=None,  y=None, R=None):
        """Return the specific dicharge components at x, and y."""
        WellBase.check_pars_presence({'R': R})
        qx, qy = WellBase.qxqy(Q, self.xw, self.yw, x, y)
        x, y = WellBase.check_xy(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y, R=R) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def rdiv(self, *args, **kwargs):
        """Return radius of water divide."""
        return np.NaN     

class wDeGlee(WellBase):
    """Axial symmetric flow to well in semi-confiende aqufier according to De Glee.
    
    This the same as Blom for r > [dd == Nc]
    
    """
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        """Return the steady state drawdown caused by a well in a semi-confined aquifer.
        
        Parameters
        ----------
        xw, yw: float
            The well's coordinates.
        rw: float
            The well's radius.
        z1, z2: float, optional
            Top and bottom elevation of the well's screen.
        aqprops: dictionary
            aquifer properties (kD, c) or (kD lambda)
        """
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD', 'c'}, self.aq)
        return
    
    def h(self, Q=None, x=None, y=None):
        """Return head in wate table aquifer.
        
        The aquifer is semi-confined and, therefore, has constant kD.
        """
        raise NotImplementedError("h not implemented for De Glee well type.")


    def dd(self, x=None, y=None, Q=None):
        """Return drawdown using uniform kD.
        
        Parameters
        ----------
        x: float, np.ndarray
            x-coordinates
        y: floar, np.ndarray, optional
            y-coordinates.
        Q: float
            The well's extraction.
        """        
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s =   FQ * K0(r / self.aq['lambda'])
        return WellBase.itimize(s)        
        
        
    def Qr(self, Q=None, x=None,  y=None):
        """Return Q(r)"""
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)
        Qr_ = Q / (2 * np.pi * self.aq['lambda']) * K1(r / self.aq['lambda'])
        return WellBase.itimize(Qr_)
            
    
    def qxqy(self, Q=None, x=None,  y=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xy(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)

class Brug370_1(WellBase):
    """Class that implements Bruggemans solution 370_01, which
    implements the steady-state drawdown due to a well in a leaky aquifer of
    which the transmissivity and the resistance of the overlying
    aquitard jump at x=0.
    """

    def __init__(self, xw=None, yw=None,  rw=0.2, z1=None, z2=None, aqprops={}):
        """Return drawdown according to Bruggeman's (1999) solution 370_01.
        
        This solution is for the steady state drawdown caused by a well in a
        semi-confined aquifer in which the properties kD and c jump at x=0.
        
        Make sure the coordinates are such that x=0 corresponds with the
        jump in the aquifer properties.
        
        Parameters
        ----------
        xw, yw: float
            The well's coordinates.
        rw: float
            The well's radius.
        z1, z2: float, optional
            Top and bottom elevation of the well's screen.
        aqprops: dictionary
            aquifer properties (k1, kD2, c1, c2)
        """
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        if not {'kD1', 'kD2', 'c1', 'c2'}.issubset(self.aq.keys()):
            raise ValueError("Missing one or more of kD1, kD2, c1, k2")

    @classmethod
    def A(cls, alpha, a, kD1, kD2, L1, L2):
        """Return function A(alpha) in Bruggeman 370_07.
        Parameters
        ----------
        alpha: integration parameter
        a: float a >= 0
            location of the well
        kD1, kD2 floats
            transmissivities for x < 0 and for x > 0
        L1, L2: floats
            L1 = np.sqrt(kD1 c1)
            L2 = np.sqrt(kD2 c2)
        """
        return np.exp(-a * np.sqrt(alpha ** 2 + 1 / L2 ** 2)) / (
            kD1 * np.sqrt(alpha ** 2 + 1 / L1 ** 2) + kD2 * np.sqrt(alpha ** 2 + 1 / L2 ** 2)
        )

    @classmethod
    def arg1(cls, alpha, a, x, y, kD1, kD2, L1, L2):
        """Return integrand in formula for phi1(x, y)"""
        return cls.A(alpha, a, kD1, kD2, L1, L2) * np.exp(x * np.sqrt(alpha ** 2 + 1 / L1 ** 2)) * np.cos(y * alpha)

    @classmethod
    def arg2(cls, alpha, a, x, y, kD1, kD2, L1, L2):
        """Return integrand for formula for phi2(x, y)"""
        return cls.A(alpha, a, kD1, kD2, L1, L2) * np.exp(-x * np.sqrt(alpha ** 2 + 1 / L2 ** 2)) * np.cos(y * alpha)

    def dd(self, x=None, y=None, Q=None):
        r"""Return drawdown problem 370_01 from Bruggeman(1999).
        
        The problem computes the steady drawdown in a well with extraction $Q$ at $xw=a, yw=0$
        in a leaky aquifer in which the properties $kD$ and $c$ jump at $x=0$ with
        $kD_1$ and $c_1$ for $x \le 0$ and $kD_2$ and $c_2$ for $x \ge 0$.
        
        Parameters
        ----------
        x, y: floats or arrays
            x, y coordinates
            Make sure that x corresponds with the jump in the aquifer properties.
        Q: float
            extraction
            
        Returns
        -------
        X, Y, Phi: a 3-tuple
            The coordinates and the head as np.ndarray of the correct shapes.
            Make sure you select the last item of the tuple to get just Phi.
        """
        # Convert scalars to arrays for vectorized computation
        
        x, y = WellBase.check_xy(x, y)
        if y is None:
            y = np.zeros_like(x)
        
        # If xw < 0, transform the problem and reuse function
        if self.xw < 0:
            # When xw < 0 just revers the x-coordinates and the properties
            aqprops = {}
            for p1, p2 in zip(['kD1', 'kD2', 'c1', 'c2'], ['kD2', 'kD1', 'c2', 'c1']):
                aqprops[p1] = self.aq[p2]
            # And then run the reversed case.
            w = Brug370_1(xw=-self.xw, yw=self.yw,  rw=self.rw,
                          z1=self.z1, z2=self.z2, aqprops=aqprops)
                
            x, y, phi = w.dd(Q=Q, x=-x, y=y)
            # Reverse x again before returning.
            return -x, y, phi

        # Compute characteristic length for $x<0$ and $x>0$
        L1 = np.sqrt(self.aq['kD1'] * self.aq['c1'])
        L2 = np.sqrt(self.aq['kD2'] * self.aq['c2'])
        
        # Create output array for head Phi
        phi = np.full_like(x, np.nan, dtype=np.float64)

        # Mask for points where x != a
        mask_x_a = x != self.xw
        mask_x_neg = x < 0
        mask_x_pos = ~mask_x_neg
        
        # Evaluate `phi` for x < 0
        if np.any(mask_x_neg):
            valid = mask_x_a & mask_x_neg
            phi[valid] = np.vectorize(
                lambda x_, y_: Q / np.pi * quad(self.__class__.arg1, 0, np.inf,
                    args=(self.xw, x_, y_, self.aq['kD1'], self.aq['kD2'], L1, L2))[0]
            )(x[valid], y[valid])

        # Evaluate `phi` for x > 0    
        if np.any(mask_x_pos):
            valid = mask_x_a & mask_x_pos
            phi[valid] = np.vectorize(
                lambda x_, y_: (
                    Q / (2 * np.pi * self.aq['kD2']) *
                    (K0(np.sqrt((x_ - self.xw) ** 2 + y_ ** 2) / L2) -
                     K0(np.sqrt((x_ + self.xw) ** 2 + y_ ** 2) / L2))
                        + Q / np.pi * quad(self.__class__.arg2, 0, np.inf,
                        args=(self.xw, x_, y_, self.aq['kD1'], self.aq['kD2'], L1, L2))[0]
                )
            )(x[valid], y[valid])

        return x, y, phi
    
    def h(self, *args, **kwargs):
        raise NotImplementedError("h not implemented for Brug370_1")
    
    def Qr(self, *args, **kwargs):
        raise NotImplementedError("Qr not implemented for Brug370_1")

    def qxqy(self, *args, **kwargs):
        raise NotImplementedError("qxqy is not implemented for Brug370_1")

class wVerruijt(WellBase):
    """Axial symmetric flow according to Verruijt and Blom.
    
    Verruijt is Dupuit + Recharge N for r <= R. So set N=0 to get Dupuit.
        
    """
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        """Return a Verruijt well object.
        
        Parameters
        ----------
        xw, yw: float
            Coordinates of the well.
        rw: float, optional.
            Well radius.
        z1, z2: float, optional
            Top and bottom elevation of the well screen.
        aqprops: dicttionary
            The aquifer propertie as needed for this well (kD, or k and D).
            To compute the object's head function, k, D must be provided
            separately in aqprops.
        """
        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'kD'}, self.aq)
        return

    def h(self, x=None, y=None, Q=None, R=None, N=None):
        """Return wetted aquifer thickness according to Verruijt."""
        WellBase.check_pars_presence({'R': R, 'N': N})
        WellBase.check_keys({'k', 'D'}, self.aq)
        x, y = WellBase.check_xy(x, y)
        if R is None or N is None:
            raise ValueError("in wVerruijt.h R and N must not be None!")
        
        r = self.radius(x, y)
        H = self.aq['D']        
        FQ = Q / (np.pi * self.aq['k'])
        
        h2 = np.zeros_like(r) + H ** 2       
        h2[r < R] = H ** 2  - FQ * np.log(R / r[r < R]) +\
            N / (2 * self.aq['k']) * (R ** 2 - r[r < R] ** 2)
        h2[h2 < 0] = 0.        
        return WellBase.itimize(np.sqrt(h2))        

    def dd(self, x=None, y=None, Q=None, R=None, N=None):
        """Return drawdown according to Verrijt."""
        WellBase.check_pars_presence({'R': R, 'N': N})
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        FQ = Q / (2 * np.pi * self.aq['kD'])
        s = np.zeros_like(r)
        s[r < R] = FQ * np.log(R / r[r < R]) - N / (4 * self.aq['kD']) * (R **2 - r[r < R] ** 2)
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, x=None, y=None, Q=None, N=None):
        """Return the flow at the distance r computed from x and y.
        
        Parameters
        ----------
        Q: float
            The well's extraction.
        N: float
            The recharge.
        x: float or np.ndarray
            x-coordinate(s)
        y: float, optional
            y-coordinate(s). Ignored if None
        """
        WellBase.check_pars_presence({'R': R, 'N': N})
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        
        Qr_ = np.zeros_like(r)
        Qr_[r <= R] = Q - np.pi * N * r[r <= R] ** 2
        
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def qxqy(self, x=None,  y=None, Q=None, R=None, N=None):
        """Return the specific dicharge components at x, and y."""
        WellBase.check_pars_presence({'R': R, 'N': N})
        x, y = WellBase.check_xy(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y, R=R, N=N) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def rhdiv(self, Q=None, R=None, N=None):
        """Return radius of water divide.
        
        The water divide exists when the head or drawdown gradient
        is zero for some r < R. This is the returned water divide.
        
        Parameters
        ----------
        Q: float
            The well's extraction.
        N: float
            The recharge.
        R: float
            The radius of fixed head or zero drawdown.
        """      
        WellBase.check_pars_presence({'R': R, 'N': N})        
        r = np.sqrt(Q / (np.pi * N))
        return r if r < R else R
    
    def rdiv(self, Q=None, R=None, N=None):
        """Return radius of water divide.
        
        The water divide exists when the head or drawdown gradient
        is zero for some r < R. This is the returned water divide.
        
        Parameters
        ----------
        Q: float
            The well's extraction.
        N: float
            The recharge.
        R: float
            The radius of fixed head or zero drawdown.
        """      
        WellBase.check_pars_presence({'R': R, 'N': N})        
        r = np.sqrt(Q / (np.pi * N))
        return r if r < R else np.nan

class wBlom(WellBase):
    """Axial symmetric flow according to Verruijt and Blom.
    
    Blom equals Verruijt for R < R[dd == Nc] and DeGlee for R >= R[dd == Nc]
    
    """
    def __init__(self, xw=0., yw=0., rw=0.2, z1=None, z2=None, aqprops={}):
        """Return a Blom well object.
        
        A Blom well applies Verruijt for r smaller than where dd > Nc and
        De Glee's semi-confined drawdown beyond that point. Blom's well
        Should be used in a well-drained area, in which the reduction of
        the groundwater ischarge to the regional drainage adds to the
        groundwater. The drawdown Nc with c the drainage resistance marks
        the distance beyond which the drainage still operates partially
        and for which at shorter distances the discharge has completely
        seized, due to dry drains and ditches.
        The transfer distance is iteratively determined by the getR method,
        which applies Newtons method to do so.
        
        Parameters
        ----------
        xw, yw: float
            Coordinates of the well.
        rw: float, optional.
            Well radius.
        z1, z2: float, optional
            Top and bottom elevation of the well screen.
        aqprops: dicttionary
            The aquifer properties as needed for this well type (k, D, c ).
            Where c is the drainage resistance [T]. (kD and c or lambda)
            is allowed as long as the h-method is not used.
            The h-method uses 'D' as Hinf, height above the bottom of
            the water table aquifer.
        """

        super().__init__(xw=xw, yw=yw, rw=rw, z1=z1, z2=z2, aqprops=aqprops)
        WellBase.check_keys({'k', 'D', 'c'}, self.aq)
        return

    def h(self, x=None, y=None, Q=None, N=None):
        """Returng h using 'k' and 'D' has Hinf."""
        WellBase.check_pars_presence({'N': N, 'Q': Q})
        x, y = WellBase.check_xy(x, y)
        r = self.radius(x, y)
        R = self.getR(Q=Q, N=N, R=r.mean())
        Hinf = self.aq['D']
        HR = Hinf - N * self.aq['c']
        
        h2 = np.zeros_like(r) + HR ** 2
        h2[r < R] = HR ** 2 + (N / (2 * self.aq['k']) * (R ** 2 - r[r < R] ** 2) 
                            - Q / (np.pi * self.aq['k']) * np.log(R / r[r < R]))
        h2[h2 <= 0] = 0.
        h = np.sqrt(h2)

        RL = R  / self.aq['lambda']
        QR = Q - np.pi * R **2 * N     
        h[r >= R] = Hinf - QR / (2 * np.pi * self.aq['kD']) *\
            K0(r[r >= R] / self.aq['lambda']) / (RL * K1(RL))
            
        h = WellBase.itimize(h)
        return h


    def dd(self, x=None, y=None, Q=None, N=None):
        """Return drawdown using uniform kD (as in a confined aquifer)."""
        WellBase.check_pars_presence({'N': N, 'Q': Q})
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)
        
        # Distance where drawdown = Nc
        R = self.getR(Q=Q, N=N, R=2 * self.rw)
        s = np.zeros_like(r)
        s[r < R] =  (Q / (2 * np.pi * self.aq['kD']) * np.log(R / r[r < R])
                - N / (4 * self.aq['kD']) * (R ** 2 - r[r < R] ** 2) + N * self.aq['c'])

        RL = R / self.aq['lambda']
        QR = Q - np.pi * R ** 2 * N        
        s[r >= R] = (QR / (2 * np.pi * self.aq['kD']) *
                     K0(r[r >= R] / self.aq['lambda']) / (RL * K1(RL)))
        s = WellBase.itimize(s)
        return s
    
    def Qr(self, x=None,  y=None, Q=None, N=None):
        """Return Q(r)"""
        WellBase.check_pars_presence({'N': N, 'Q': Q})
        x, y = WellBase.check_xy(x, y)
        
        r = self.radius(x, y)

        R = self.getR(r=r.mean(), Q=Q, N=N)
        QR = Q - np.pi * R ** 2 * N        
        R_lam = R / self.aq['lambda']
        r_lam = r / self.aq['lambda']
        Qr_ = np.zeros_like(r)
        Qr_[r < R]  = Q - np.pi * r ** 2 * N
        Qr_[r >= R] = QR * r_lam[r >= R] * K1(r_lam[r >= R]) / K1(R_lam)
        Qr_ = WellBase.itimize(Qr_)
        return Qr_
    
    def q(self, x=None, y=None, Q=None, N=None):
        """Return the recharge q."""
        r = self.radius(x, y)
        R = self.getR(R=r.mean(), Q=Q, N=N)        
        q = np.zeros_like(r)
        q[r <= R] = N
        q[r >  R] = N * K0(r[r > R] / self.aq['lambda']) / K0(R / self.aq['lambda'])
        return WellBase.itimize(q)

    
    def qxqy(self, x=None,  y=None, Q=None):
        """Return the specific dicharge components at x, and y."""
        x, y = WellBase.check_xy(x, y)
        if y is None:
            y = np.zeros_like(x)     
        r = self.radius(x, y)
        qr = self.Qr(Q, x, y) / (2 * np.pi * r)
        qx = qr * (self.xw - x) / r
        qy = qr * (self.yw - y) / r
        qx = WellBase.itimize(qx)
        qy = WellBase.itimize(qy)
        return (qx, qy)
        
    def rdivh(self, Q=None, N=None):
        """Return location of water divide (freatic)."""
        R = self.getR(Q=Q, N=N, R=1.0)
        rD = np.sqrt(Q / (2 * np.pi * N))
        return rD if rD <= R else np.nan
    
    def rdivD(self, Q=None, N=None):
        """Return location of water divide (fixed D).
        
        In the case of Blom, there can never be a divide.
        If there is extraction (Q > 0) the devide is at infinity.
        When there is injection the devide might be considered the well face.
        Always return np.nan
        """
        R = self.getR(Q=Q, N=N, R=1.0)        
        rD = np.sqrt(Q / (np.pi * N))
        return rD if rD <= R else np.nan
    
    def y(self, R=1.0, Q=None, N=None):
        """Return y for Newton Raphson detetermination of R."""
        QR = Q - np.pi * R ** 2 * N
        Rl = R  / self.aq['lambda']
        return -N * self.aq['c'] + QR / (2 * np.pi * self.aq['kD']) * K0(Rl) / (Rl * K1(Rl))
    
    def y1(self, R=None, Q=None, N=None):
        """Return dy(R)/dR for Newton Raphson determination of R."""        
        RL = R / self.aq['lambda']
        k0k1 = K0(RL) / K1(RL)        
        return -N * self.aq['lambda'] / self.aq['kD'] * k0k1 -\
            (Q / R / (2 * np.pi * self.aq['kD'])  - N * R / (2 * self.aq['kD'])) *\
                (1 - k0k1  ** 2)

    # Newton Raphson functions
    def dydR(self, R=None, Q=None, N=None, dr=0.01):
        """Return numerical derivative of y in Newton iterations."""
        return (self.y(R=R + dr, Q=Q, N=N) -
                self.y(R=R - dr, Q=Q, N=N)) / (2 * dr)
                
    def getR(self, R=1.0, Q=None, N=None, tolR=0.1, n=50,  verbose=False):
        """Return R by Newton's method using the input's R as starting value.
        
        The function of which the zero is to be found is mooth, so that
        the Newton method should always work.
        
        Parameters
        ----------
        Q: float
            Well extraction Q > 0).
        N: float
           Recharge effective in the Verruit part (r < R-final)
        R: float
          intial R to start the Newton search process.
        tolR: float
            Accuracy used to break the iteration process.
            Because we search for the real-world distance beyond
            the Verruijt portion changes to semiconfined deGlee
            portion, tolR =0.1 seems accurate enough.
        n: int
           Maximum number of Newton iterations.        
        """
        if verbose:
            print(f"R initial={R:.3g} m")
        if Q < 0:
            print(f"Warning: Q < 0 = {Q:.4g} [m3/d]; abs value is used instead.")
            Q = abs(Q)
        for i in range(n):
            dR = -self.y(R=R, Q=Q, N=N) / self.y1(R=R, Q=Q, N=N)
            #dR = -self.y(R=R, Q=Q, N=N) / self.dydR(R=R, Q=Q, N=N)
            R += dR
            if verbose:
                print(f"iteration {i}, R={R:.3g} m, dR = {dR:.3g} m")
            if abs(dR) < tolR:
                if verbose:
                    print(f"R final ={R:.3g} m")
                self.R = R
                return R
        R = np.nan
        self.R = R        
        print(f"R final ={R:.3g} m")
        return R
    
    def plot_newton_progress(self, R=None, Q=None, N=None, figsize=(10, 6)):
        """Plot progress of the Newton method to find R at which the drawdown is Nc.
        
        Parameters
        ----------
        R: float
            initial R to start iteration.
        Q: float
            well extraction
        N: float
            recharge
        """
        r = np.linspace(0, 300, 101)[1:]
        title = f"y(R) and progress of Newton iterations for Q={Q:.4g} , N={N:.4g}, c={self.aq['c']:.4g}"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=title, xlabel="r [m]", ylabel="y(r) + Nc [m]")
        
        Nc = N * self.aq['c']
        ax.plot(r, self.y(R=r, Q=Q, N=N) + Nc, label='y(r) + Nc')
        
        for _ in range(20):
            #y, y1 = self.y(R=R, Q=Q, N=N), self.dydR(R=R, Q=Q, N=N)     
            y, y1 = self.y(R=R, Q=Q, N=N), self.y1(R=R, Q=Q, N=N)
            ax.plot([R, R], [Nc, y + Nc], 'k')     
            ax.plot(R, y + Nc, 'bo', mfc='none')
            dR = - y / y1
            ax.plot([R, R + dR], [y + Nc, Nc], 'k')            
            R += dR
            if abs(dR) < 0.1:
                break
        ax.grid()
        ax.plot(ax.get_xlim(), (Nc, Nc), 'k--', label=fr"$N c = {Nc}\, m$")          
        ax.legend()
        return ax
        
    def plot_derivative_of_y(self, R=None, Q=None, N=None, figsize=(10, 6)):
        """Plot the derivative of y, analytical and numerically calculated."""
        r = np.linspace(0, 300, 301)[1:]        
        title = f"Derivatives of y for Q={Q:.4g} , N={N:.4g}, c={self.aq['c']:.4g}"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set(title=title, xlabel="R [m]", ylabel="y(r) [m]")
        ax.plot(r, self.y1(  R=r, Q=Q, N=N), '-', label='y1 (analytic)')        
        ax.plot(r, self.dydR(R=r, Q=Q, N=N), '.', label='dydR (numerical)')
        ax.legend()
        return ax

class StripBase(ABC):
    """Base class for several 1D groundwater flow functions."""
        
    def __init__(self, aqprops={}):
        """Initialyze a well.
        
        Parameters
        ----------
        aqprops: dict
            Aquifer propertiers (see below)
            
        """
        self.aq = aqprops

    @abstractmethod
    def h(self, *args, **kwargs):
        """Return the drawdown."""
        pass

    @abstractmethod
    def dd(self, *args, **kwargs):
        """Return the drawdown."""
        pass

    @abstractmethod
    def Qx(self, *args, **kwargs):
        """Return the dicharge at x values."""
        pass
        
    @staticmethod
    def check_x(x=None):
        """Return x as float array for values >= 0"""       
        x = np.atleast_1d(x).astype(float)
        return x
        
    @staticmethod
    def check_xt(x=None, t=None):
        """Return x, t making sure x is sclar if t is array and vice versa."""
        if isinstance(t, np.ndarray):
            t = t[t >= 0]
            if not np.isscalar(x):
                raise ValueError("x must be scalar if t is an array!")
        else:
            x = StripBase.check_x(x)
        return x, t
    
class Section(StripBase):
    """This class implements the head or drawdown in a cross section that between xL and xR
    is a water table aquifer or a confined aquifer with recharge. Outside, i.e. for x < xL
    and x > xR the aquifer is semi.confined. The flow and the head are both continuous
    at x = xL and x=xR. This solution is meant to simulate a cross section through 
    a high and dry area bounde on both sides by a low marshy, well-drained area.
    Only steady-state flow is considered.
    """
    
    def __init__(self, boundaries={}, aqprops={}):
        """Return a Section object.
        
        The aquifer section has uniform kD.
        For xL < x< xR it has recharge N.
        For x < xL and x > xR the aquifer is leaky with the given values for lamb_L and lamb_R.
        For fixed heads at xL and xR set lamb_L and lamb_R to zero and the
        head at xL and xR will be hL and hR respectively.

        This head, therefore shows the head in a strip of land between xL and xR
        bounded on both sides by marshy or well-drained areas characterized by
        their characteristic length lambda, which can be different at the left and
        at the right.

        Parameters
        ----------
        boundaries: dictionary {xL, xR, hL, hR}
            physical boundaries of the section.
        aqprops: dictionary {kD, lambda_L, lambda_R}
            aquifer properties
        """        
        self.aq = aqprops
        if {'k', 'D'}.issubset(aqprops):
            aqprops.update(kD=aqprops['k'] * aqprops['D'])
        if not {'kD', 'lambda_L', 'lambda_R'}.issubset(aqprops):
            raise ValueError("kD, lambda_L and or Lambda_R missing in aqprops!")
        
        self.bnd = boundaries
        if not {'xL', 'xR', 'hL', 'hR'}.issubset(boundaries):
            raise ValueError("xL, xR, hL and or hR missing in boundaries dict.")
        return 
            

    def h(self, x=None, N=None):
        """Return head for the given x-values.
        
        Parameters
        ----------
        x: np.ndarray or float
            Coordinate(s) where head is computed. x may extend beyond xL and xR.
        N: float
            Recharge.
        """
        eps = 1e-6
        xL, xR = self.bnd['xL'], self.bnd['xR']
        hL, hR = self.bnd['hL'], self.bnd['hR']
        kD, lamb_L, lamb_R = self.aq['kD'], self.aq['lambda_L'], self.aq['lambda_R']
        L = xR - xL        
        
        # Flow at left-hand area boundary (at x=xL)
        QL = self.Qx(x=xL, N=N)
        QR = QL + N * L
                
        # Head above hL and HR caused by QL and QR respectively
        dhR = +QR  * lamb_R / kD
        dhL = -QL  * lamb_L / kD
        
        x = StripBase.check_x(x)
        h = np.zeros_like(x)
        
        # Deal with the three area separately
        mask_L = x < xL
        mask_R = x > xR
        mask_C = ~(mask_L | mask_R)
        
        # Center area
        h[mask_C] = (hL - QL * lamb_L / kD - QL / kD * (x[mask_C] - xL)
                    - N / (2 * kD) * (x[mask_C] - xL) ** 2)
        # Left area
        h[mask_L] = hL + dhL * np.exp(+(x[mask_L] - xL) / (lamb_L + eps))
        # Right area
        h[mask_R] = hR + dhR * np.exp(-(x[mask_R] - xR) / (lamb_R + eps))
        
        return WellBase.itimize(h)


    def dd(self, x=None, N=None):
        """Return head change for the given x-values.
        
        Parameters
        ----------
        x: np.ndarray or float
            Coordinate(s) where head is computed. x may extend beyond xL and xR.
        N: float
            Recharge.
        """
        raise NotImplementedError("Method dd not implemented, it's not useful here, use method h(x, N) instead")


    def Qx(self, x=None, N=None):
        """Return the discharge Qx for given x-values.
        
        Parameters
        ----------
        x: np.ndarray or float
            Coordinate(s) where head is computed. x may extend beyond xL and xR.
        N: float
            Recharge.        
        """
        eps = 1e-6
        xL, xR, hL, hR = self.bnd['xL'], self.bnd['xR'], self.bnd['hL'], self.bnd['hR']
        kD, lamb_L, lamb_R = self.aq['kD'], self.aq['lambda_L'], self.aq['lambda_R']
        L = xR - xL
            
        # Discharge at xL
        QL = (-kD / (L + lamb_L + lamb_R)  * (hR - hL + N / (2 * kD) * L ** 2 +
                                            N / kD * L * lamb_R))
        QR = QL + N * L
        
        x = StripBase.check_x(x)
        Q = np.zeros_like(x)
        
        # Deal with the three area separately
        mask_L = x < xL
        mask_R = x > xR
        mask_C = ~(mask_L | mask_R)

        Q[mask_L] = QL * np.exp(+(x[mask_L] - xL) / (lamb_L + eps))
        Q[mask_C] = QL + N * (x[mask_C] - xL)
        Q[mask_R] = QR * np.exp(-(x[mask_R] - self.bnd['xR']) / (lamb_R + eps))
        return WellBase.itimize(Q)

class Verruijt1D(StripBase):
    """Verruijt 1D solution.
    
    The assumptions are the same as for the axial symmetric Verruijt solution.
    Hence, the extraction is at x=0, while head is unaltered (drawdown = 0) at
    x = L. It's a cross section with steady state flow, for x >= 0 with
    recharge on top.
    """    
    def __init__(self, aqprops={}):
        """Return a 1D Verruijt object.
        
        This object is similar to the well-known 2D variant. It is
        drawdown caused by extraction with the effect of superimposed,
        while a fixed head (zero drawdown) is maintained at a given distance L.
        Hence, for 0 < x < L there may be a water divide (zero gradient).
        
        For N=0 this is the same as a 1D Dupuit.
        """
        super().__init__(aqprops=aqprops)
        
        WellBase.check_keys({'kD'}, self.aq)
        return


    def dd(self, Q=None, x=None, L=None, N=None):
        """Return drawdown (using confined kD).
        
        Parameters
        __________
        Q: float [L2/T]
            extraction at x=0.
        x: float or np.ndarray
            x-coordinate(s).
        L: float
            The x where drawdown is 0
        N: float [L/T]
            The recharge.
        """
        x = super().check_x(x)
        s = np.zeros_like(x)
        s[x < L] = (Q / self.aq['kD'] * (L - x[x < L]) -
                    N / (2 * self.aq['kD']) * (L **2 - x[x < L] ** 2))
        if len(s) == 1:
            s = s.ravel()[0]
        return s
    
    def Qx(self, Q=None, x=None, L=None, N=None):
        """Return Qx.
        
        Parameters
        ----------
        Q: float [L2/T]
            The extraction at x=0.
        x: float or np.ndarray
            x-coordinate(s).
        L: float
            The x where drawdown is 0
        N: float [L/T]
            The recharge.
        """
        x = super().check_x(x)
        Qx_ = np.zeros_like(x)
        Qx_[x < L] = Q - N * x[x < L]
        Qx_ = WellBase.itimize(Qx_)
        return Qx_
    
    def h(self, Q=None, x=None, L=None, N=None):
        """Return head using unconfined k and D."""
        WellBase.check_keys({'k', 'D'}, self.aq)
        x = np.atleast_1d(x).astype(float)
        Hinf = self.aq['D']
        h = np.zeros_like(x) + Hinf
        h[x < L] = np.sqrt(Hinf ** 2  + N / self.aq['k'] * (L ** 2 - x[x < L] ** 2) -
                    2 * Q / self.aq['k'] * (L - x[x < L]))
        h = WellBase.itimize(h)
        return h
    
    def xdiv(self, Q=None, L=None, N=None):
        """Return location of water divide."""
        x = - Q / N
        if x > L:
            return np.nan
        else:
            return Q / N
        
class Blom1D(StripBase):
    """Class for Blom's solutions for 1D flow.
    
    Blom has Verruijt for x < x[dd == Nc] and mazure for x > x[dd == Nc]
    Mazure is steady state flow into or from a leaky aquifer with
    zero head on top and a given head at x=0. h(x) = h(0) exp(- x/lambda)
    
    It is meant to simulate an area (x>0) where for smaller x values
    the drawdown is high enough to stop drainage in which the recharge
    feeds the aquifer instead of flowing to the drainage ditches and drains,
    while for larger distance, the drawdown is not sufficient to capture
    al recharge, where some of the recharge is still drained.
    The boundary between the two zones is where the drawdown equals Nc
    with N the recharge and c the drainage resistance.
    """
    def __init__(self, aqprops={}):
        """Return a Blom1D object.
        
        Parameters
        ----------
        aqprops: dictonary
            aquifer properties {'kD', 'c'} or {'k', 'D', 'c'}
        """
        super().__init__(aqprops=aqprops)
        
        WellBase.check_keys({'kD', 'c'}, self.aq)
        return

    def getL(self, Q=None, N=None):
        """Return L where drawdown is Nc."""
        if Q < 0:
            print(f"Warning, Q < 0,  {Q:.4g} m2/d, abs value is used.")
            Q = abs(Q)
        L = Q / N - self.aq['lambda']
        L = L if L > 0. else 0.
        return L
        
    def h(self, Q=None, x=None, N=None):
        """Return h using for unconfined aquifer 'k' and 'D' as Hinf."""
        WellBase.check_keys({'k', 'D', 'c'}, self.aq)        
        Hinf = self.aq['D']
        HL = self.aq['D'] - N * self.aq['c']
        L = self.getL(Q=Q, N=N)
        h2 = HL ** 2 + N / self.aq['k'] * (L ** 2 - x ** 2) - 2 * Q / (self.aq['k']) * (L - x)
        if np.isscalar(x):
            if x > L:
                return Hinf - N * self.aq['c'] * np.exp(-(x - L) / self.aq['lambda'])
            else:
                return np.sqrt(h2)
        h = np.zeros_like(x)
        h[x < L]  = np.sqrt(h2[x < L])
        h[x >= L] = Hinf - N * self.aq['c'] * np.exp(-(x[x >= L] - L) / self.aq['lambda'])
        h = WellBase.itimize(h)
        return h

    def q(self, Q=None, x=None, N=None):
        """Return the recharge q."""
        x = super().check_x(x)
        L = self.getL(Q=Q, N=N)
        q = np.zeros_like(x)
        q[x < L] = N
        q[x >= L] = N * np.exp(-(x[x >= L] - L) / self.aq['lambda'])
        return WellBase.itimize(q)

    def dd(self, Q=None, x=None, N=None):
        """Return drawdown using uniform 'kD' of aquifer."""
        x = super().check_x(x)
        L = self.getL(Q=Q, N=N)
        s = np.zeros_like(x)
        s[x < L] = (Q / self.aq['kD'] * (L - x[x < L]) -
                N / (2 * self.aq['kD']) * (L **2 - x[x < L] ** 2) + N * self.aq['c'])
        s[x >= L] = N * self.aq['c'] * np.exp(-(x[x >= L] - L) / self.aq['lambda'])
        s = WellBase.itimize(s)
        return s
        
    def Qx(self, Q=None, x=None, N=None):
        """Return Q(x)."""
        x = super().check_x(x)
        L = self.getL(Q=Q, N=N)
        Qx_ = np.zeros_like(x)
        Qx_[x < L] = Q - N * x[x < L]
        Qx_[x >= L] = N * self.aq['lambda'] * np.exp(- (x[x >= L] - L) / self.aq['lambda'])
        Qx_ = WellBase.itimize(Qx_)
        return Qx_
        
    def xdiv(self, Q=None, N=None):
        """Return x location of water divide."""
        L = self.getL(Q=Q, N=N)
        if Q / N < L:
            return L
        else:
            return np.nan
        

def ground_surface(x, xL=None, xR=None, yM=1, Lfilt=20, seed=3):
    r"""Return a ground surface elevation for visualization purposes.
    
    The elevation will be 0 at xL and xR and approach yM in the middle.
    
    Random numbers are used that are smoothed by filfilt using integer Lfilt
    as the filter width, and are finally multiplied
    
    $$ y_M \times np.sqrt{\cos \left(\pi \frac{x - xM}{L}\right)}$$
    
    where $yM$ is the maximum elevation.

    Lfilt can be adapted to the number points in x.

    Parameters
    ----------
    x : ndarray
        Coordinates.
    xL, xR : floats, optional
        Left and right coordinate bounds for the surface. Defaults to first and last x values.
    yM : float, optional
        Approximate maximum height of the surface before smoothing. Default is 1.
    Lfilt : int, optional
        Length of the smoothing filter applied with scipy.signal.filtfilt. Default is 20.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ndarray
        Smooth ground surface elevation.
    
    @TO 2025-02-10
    """
    # The points of zero elevation, xL and xR are optional
    if xL is None:
        xL = x[0]
    if xR is None:
        xR = x[-1]
        
    L, xM = xR - xL, 0.5 * (xL + xR)
    
    y = yM * np.sqrt(np.abs(np.cos(np.pi * (x - xM) / L)))

    # Ensure Lfilt is at least 2 to avoid issues with filtfilt
    Lfilt = max(Lfilt, 2)

    # Random noise generation with optional seed
    rng = np.random.default_rng(seed)
    z = rng.random(len(x))

    # Apply smoothing and return elevation (method 'gust' prevents padding problems)
    filtered = y * filtfilt(np.ones(Lfilt) / Lfilt, 1, z, method='gust')
    return filtered


class Mirrors():
    """Class to compute mirror ditches plus signs or mirror wells plus signs
    
    In case of ditches the position of the ditches are returned with their signs.
    In case of a well, the mirror well positions are returned.
    """
    def __init__(self, xL, xR, xw=None, N=30, Lclosed=False, Rclosed=False):
        """Return x-coordinate of mirror points given a strip between xL and xR, xp of point of well.

        Parameters
        ----------
        xL, xR: floats
            x-coordinates of left and right of strip between the two head boundaries
        N: int
            number of mirror wells positions
        """
        if xL > xR:
            xL, xR = xR, xL
        self.xw = xw
        self.xL = xL
        self.xR = xR
        self.Lclosed = Lclosed
        self.Rclosed = Rclosed
        self.N = N
        
        # The actual well (May be None for just mirroring ditches)
        self.xw = xw
        self.sw = 1

        # Starting values for the coordinates of the mirrors
        # The right ditch (xR) and the left ditch (xL)
        xRD, xLD = [xR], [xL]
        
        # first mirror wells signs are opposite to self.sw
        if self.xw is not None:
            sRD = [self.sw if Rclosed else -self.sw]
            sLD = [self.sw if Lclosed else -self.sw] 
        
        # No wells, just ditches. First ditches have positive sign
        # that may be inverted later based on Lclosed and Rclosed
        else:
            sRD, sLD = [1], [1]

        # Get mirror ditch or well coordinates starting with
        # either the right ditch (xRD) or with the left  ditch (xLD)
        for i in range(1, N):
            if i % 2 == 1:
                xRD.append(xL - (xRD[-1] - xL))
                xLD.append(xR - (xLD[-1] - xR))
                sRD.append(sRD[-1] if Lclosed else -sRD[-1])
                sLD.append(sLD[-1] if Rclosed else -sLD[-1])
            else:
                xRD.append(xR - (xRD[-1] - xR))
                xLD.append(xL - (xLD[-1] - xL))
                sRD.append(sRD[-1] if Rclosed else -sRD[-1])
                sLD.append(sLD[-1] if Lclosed else -sLD[-1])
                
        self.xLD = xLD
        self.xRD = xRD
        self.sLD = sLD
        self.sRD = sRD
        
        # If a well position was given we don't want the mirror ditches but the mirror wells:
        if self.xw is not None:
            xM = 0.5 * (xR + xL)
            deltaR = xR - xw     # mirror position over right ditch
            deltaL = xL - xw     # mirror position over left  ditch
            for i in range(len(xRD)):
                if self.xRD[i] > xM:
                    self.xRD[i] += deltaR
                else:
                    self.xRD[i] -= deltaR
            for i in range(len(xLD)):
                if self.xLD[i] < xM:
                    self.xLD[i] += deltaL
                else:
                    self.xLD[i] -= deltaL
        return

    def show(self, ax=None, figsize=(10, 6), fcs=('yellow', 'orange')):
        """Return picture of the ditches and the direction of the head change.
        
        Parameters
        ----------
        ax: matplotlib.Axes.axes or None
            axis to plot on
        fcs: tuple  of two strings
            colors of the mirror strips and the central strips respectively
        """
        L = self.xR - self.xL
        
        if not ax:
            _, ax = plt.subplots(figsize=figsize)   

        # Draw arrows for xLD ditches
        for x, sgn in zip(self.xLD, self.sLD):
            if sgn > 0:
                y1, y2 = sgn, 0   # upward arrow
            else:
                y1, y2 = 0, -sgn  # downward arrow
            ax.annotate("", xy=(x, y1), xytext=(x, y2),
                        arrowprops=dict(arrowstyle="->", color="red"))
            
        # Draw arrow for xRD ditches
        for x, sgn in zip(self.xRD, self.sRD):
            if sgn > 0:
                y1, y2 = sgn , 0
            else:
                y1, y2 = 0, -sgn
            ax.annotate("", xy=(x, y1), xytext=(x, y2),
                        arrowprops=dict(arrowstyle="->", color="blue")) 

        # Plot the strips and accentuate the central one
        for xl in np.sort(np.hstack((self.xL - np.arange(self.N) * L, self.xR + np.arange(self.N - 1) * L))):
            xr = xl + L
            fc = fcs[1] if xl < 0 and xr > 0 else fcs[0]        
            p = Path(np.array([[xl, xr, xr, xl, xl], [0, 0, -1, -1, 0]]).T, closed=True)
            ax.add_patch(PathPatch(p, fc=fc, ec='black'))

        ax.plot(0.5 * (self.xL + self.xR), 0, 'ro')
        
        if self.xw is not None:
            ax.annotate("", xy=(self.xw, self.sw), xytext=(self.xw, 0),
                        arrowprops=dict(arrowstyle="->", color="green")) 
        
        ax.set_title(f"Mirror ditches with     Left: {'closed' if self.Lclosed else 'open'}, Right: {'closed' if self.Rclosed else 'open'}")
        ax.set_ylim(-1, 1.1)
        
        # Don't want no yticks and no ticklabels
        ax.set_yticks([])
        return ax
    

# === E X A M P L E S ===

# %%
if __name__ == '__main__':
    # %% [markdown]
    ### Theis and Hantush type curves
    # %%
    fig, ax = plt.subplots(figsize=(10, 6))
    title = "Theis and Hantush's type function for drawdown caused by well"               
    ax.set(title=title, xlabel="1/u", ylabel="well function values",
           xscale="log", yscale="log")
    ax.grid()
    u = np.logspace(-4, 1, 51)
    ax.plot(1/u, Wt(u), '.', label="Theis")
    for rho in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1., 2., 3., 4., 5.]:
        ax.plot(1/u, Wh(u, rho), label=f"rho = {rho}")
    ax.legend(loc='lower right', fontsize='small')
    # plt.show()

    # %% [markdown]
    ### Example, Use of mirror wells, which immediately shows all 4 possibilities
    # The problem is an aquifer between xL and xR with as boundaries that the aquifer
    # at xL and or xR is fixed or closed.
    
    # %%
    # Set aquifer properties
    kD, S, c= 600, 0.001, 200
    lambda_ = np.sqrt(kD * c)
    
    # Set system extension properties and coordinates
    xL, xR, xw, Nmirror= -200, 200, 100, 30
    Lclosed, Rclosed = False, False # right and or left size of strip closed?
    L = xR - xL
    n = 0
    x = np.linspace(xL - n * L, xR + n * L, 1201)
    
    # Set extraction
    Q = -1200.

    # Show result of well between two fixed-head boundaries (using mirror wells)
    md = Mirrors(xL, xR, xw=xw, N=Nmirror, Lclosed=Lclosed, Rclosed=Rclosed)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    ax1 = md.show(ax=ax1, figsize=(10, 3), fcs=('yellow', 'orange'))

    s = np.zeros_like(x)
    
    # The drawdown of the well itself, which is at md.xw with sign md.sw
    r = np.sqrt((md.xw - x) ** 2)
    s = md.sw * Q / (2 * np.pi * kD) * K0( r / lambda_)
    
    # Add the drawdown due to mirror wells starting with the left one
    for xw, sgn in zip(md.xLD, md.sLD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)
        
    # Add the drawdown due to mirror well starting with the right one
    for xw, sgn in zip(md.xRD, md.sRD):
        r = np.sqrt((xw - x) ** 2)
        s += sgn * Q / (2 * np.pi * kD) * K0(r / lambda_)

    ax2.set_title(f"Verlaging door put met Q={Q} m3/d op xw={md.xw} m tussen randen met vaste $h$\n" +
    fr"op xL={xL} en xR={xR} m. kD={kD} m2/d, $\lambda$={lambda_:.0f} m, {Nmirror} maal gespiegeld.",
                 fontsize=10)
    ax2.set(xlabel="x [m]", ylabel='head [m]')
    ax2.plot(x, s, label=f"Q = {Q} m3/d")
    ax2.plot(xL, 0, 'ro', label='linker rand vast op 0.')
    ax2.plot(xR, 0, 'bo', label='rechter rand vast op 0.')
    ax2.grid()
    ax2.legend()
    # plt.show()
    
    # %% [markdown]
    ### Example of using mirror ditches to simulate transient filling of a basin
    
    # %%

    # Aquifer properties and coordinates
    kD, S = 600., 0.2
    xL, xR, AL, AR = -100., 100., 1.0, 1.0
    x = np.linspace(xL, xR, 201)
    
    b = (xR - xL) / 2 # Half width of the basin
    
    # Times for the simulation
    T50 = 0.28 * b ** 2 * S / kD # Halftime of the drainage of the basin
    ts = np.arange(7) * T50 # times to show
    ts[0] = 0.01 * T50 # First time should be > zero
    
    # Using xw=None in the call of Mirrors generates coords of mirror ditaches
    md = Mirrors(xL, xR, xw=None, N=30, Lclosed=False, Rclosed=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title(f"Basin kD={kD} m2/d, S={S}, T50={T50:.4g} d")
    ax.set(xlabel="x [m]", ylabel="head [m]")
    
    for t in ts:
        s = np.zeros_like(x) # Initialize
        
         # Use xM to see what ditch coordinates are to the left
         # and what are to the right of the center of basin.
        xM = 0.5 * (xL + xR)
        
        # Ditches that started with the left one
        for xD, sgn in zip(md.xLD, md.sLD):            
            x_ = xD - x if xD >= xM else x - xD
            u = x_ * np.sqrt(S / (4 *  kD * t))
            s += sgn * AL * erfc(u)
            
        # Ditches that started with the right one
        for xD, sgn in zip(md.xRD, md.sRD):
            x_ = xD - x if xD >= xM else x - xD
            u = x_ * np.sqrt(S / (4 *  kD * t))
            s += sgn * AR * erfc(u)
            
        ax.plot(x, s, label=f"t = {t:.3f} = {t/T50:.4g} T50d")
    ax.grid()
    ax.legend(loc="lower right")
    # plt.show()


    # %%[markdown]
    ### Testing the function "ground_surface"
    
    # %%
    xL, xR = -2500., 2500.
    x = np.linspace(-4000., 4000., 801)
    
    # Use yM to set max elevation (approximately, adapt by trial and error)
    yM = 2.
    
    # Filter length in filtfilt. Adaept by trial and error to length of x
    Lfilt = 20
    
    # Set seed to an integer to get the same results each time.
    # Adept by trial and error.
    seed = 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(
        f"Generated ground surface for xL={xL}, xR={xR}, yM={yM}, Lfilt={Lfilt}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("Elevation [m]")
    
    h_grsurf = ground_surface(x, xL=xL, xR=xR, yM=yM, Lfilt=Lfilt, seed=1)
    
    ax.plot(x, h_grsurf, label=f"seed={seed}")
    ax.legend()
    ax.grid()
    # plt.show()

    # %% [markdown]
    ### Head in strip with adjacent leaky aquifers
    
    # %%
    # Parameters for the cross section
    N, kD, xL, xR, hL, hR = 0.001, 600, -2500, 2500, 20, 22
    lamb_L, lamb_R = 100., 300.,
    boundaries = {'xL': xL, 'xR': xR, 'hL': hL, 'hR': hR}
    aqprops = {'kD': kD, 'lambda_L': lamb_L, 'lambda_R': lamb_R}
    
    sec1 = Section(boundaries=boundaries, aqprops=aqprops)
    sec2 = Section(boundaries=boundaries, aqprops={'kD': kD, 'lambda_L': 0., 'lambda_R': 0.})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Head in high area bounded by drained areas\n" +
                fr"kD={sec1.aq['kD']:.4g} m2/d, N={N:.4g} m/d, $\lambda_L$={sec1.aq['lambda_L']:.4g} m, $\lambda_R$={sec1.aq['lambda_R']:.4g} m")
    ax.set(xlabel="x [m]", ylabel="TAW [m]")
    ax.grid()

    ax.plot(x, ground_surface(x, xL=xL, xR=xR, yM=1, Lfilt=8) + sec1.h(x, N), label='maaiveld')

    # Case with soft center-area boundaries
    ax.plot(x, sec1.h(x, N), 
            label=f"h with hare area boundaries. N={N} m/d, kD={sec1.aq['kD']} m2/d, op xL={xL} m en xR={xR} m")

    # Case with hard center-area boundaries
    ax.plot(x, sec2.h(x, N),
            label=fr"h with leaky adjacent areas. N={N} m/d, kD={sec2.aq['kD']} m2/d $\lambda_L$={sec2.aq['lambda_L']} m, $\lambda_R$={sec2.aq['lambda_R']} m")

    # Plot the boundary locations
    # Compute the head at the boundary locations
    hxL = sec1.h(xL, N)
    hxR = sec1.h(xR, N)
    
    ax.plot([xL, xR], [hxL, hxR], 'go',
            label=fr"Boundary be with leaky area with $\lambda_L$={sec1.aq['lambda_L']} en $\lambda_R$={sec1.aq['lambda_R']} m")
    
    # Annotate boundaries
    ax.annotate("Area boundary", xy=(xL, 20.), xytext=(xL, 24.5), ha='center',
                arrowprops={'arrowstyle': '-'})
    ax.annotate("Area boundary", xy=(xR, 20), xytext=(xR, 24.5), ha='center',
                arrowprops={'arrowstyle': '-'})

    # Mark the head at both boundaries with a red dot
    hxL = sec2.h(xL, N)
    hxR = sec2.h(xR, N)
    ax.plot([xL, xR], [hxL, hxR], 'ro', label=f"Hard boundary hL={sec2.bnd['hL']}, hR={sec2.bnd['hR']} m op  resp. x={sec2.bnd['xL']} en x={sec2.bnd['xR']}  m")



    # %% [markdown]
    ### Verruijt 1D fixed Q and varying L
    # Example with ever increasing distance L to the fixed head boundary

    # %%
    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 0.5, 'N': 0.003, 'L': 250}
    x = np.linspace(0, 300, 301)[1:]
    fixed_head_distances = [50., 100., 150., 200., 250., 300.]
    fhd = fixed_head_distances

    V1 = Verruijt1D(aqprops)
    h = V1.h(x=x, **pars)
    # print(h)

    # Head
    title = f"Verruijt 1D, unconfined, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}\n" +\
        f"The distance to the fixed head boundary increases from {fhd[0]} to {fhd[-1]} m" 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=title, xlabel="x [m]", ylabel="head [m]")
                
    clrs = cycle('rbgmkcy')
    for L in fixed_head_distances:
        clr = next(clrs)
        pars['L'] = L
        h = V1.h(x=x, **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        
        # plot both the head and the head mirrord around x=0.
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        hd = V1.h(x=xd, **pars)
        ax.plot([-xd, xd], [hd, hd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [aqprops['D'], aqprops['D']], '.',
                color=clr, label=f"L={pars['L']}")

    ax.legend(fontsize=6, loc='lower right')

    
    # Drawdown
    title = f"Verruijt 1D, confined, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}\n" +\
        f"The distance to the fixed head boundary increases from {fhd[0]} to {fhd[-1]} m" 
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=title, xlabel="x [m]", ylabel="drawdown [m]")
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for L in fixed_head_distances:
        clr = next(clrs)
        pars['L'] = L    
        dd = V1.dd(x=x, **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), V1.dd(x=x, **pars), color=clr, label=label)

        xd = V1.xdiv(**pars)
        ddnd = V1.dd(x=xd, **pars)
        ax.plot([-xd, xd], [ddnd, ddnd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [0., 0.], '.', color=clr, label=f"L={pars['L']}")

    ax.legend(fontsize=6, loc='lower right')

    # plt.show()


    # %% [markdown]
    # ## Voorbeeld Verruijt 1D; vaste L, variërende Q
    # 
    # De onttrekking en, daarmee de verlaging is stapsgewijs vergroot. Hierdoor komt de waterscheiding steeds verder weg te liggen tot deze de vaste rand $L$ bereikt.
    # 
    # Bij verruijt is buiten de vaste rand geen stroming en dus ook geen verlaging.
    # 
    # Bij grote onttrekking is de daling van de grondwaterstand in het eeste plaatje is een stuk groter dan de verlaging in het tweede plaatje. Dit is het gevolg van de afname van de dikte van het watervoerende pakket waar in het eerste plaatje wel en in het tweede plaatje geen rekening mee is gehouden.

    # %% [markdown]
    ### Verruijt 1D, fixed L and varying Q

    # %%
    pars['L'] = 600
    x = np.linspace(0, 1000, 5001)[1:]
    Qs = [0., 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # head
    title = f"Blom 1D, unconfined k={aqprops['k']:.1f}, D={aqprops['D']:.1f}\n" +\
        f"The Extraction Q increases from {Qs[0]} to {Qs[-1]} m2/d" 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=title, xlabel="x [m]", ylabel="head [m]")

    clrs = cycle('rbgmkcy')
    for Q in Qs:
        clr = next(clrs)
        pars['Q'] = Q
        label=f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = V1.h(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        hd = V1.h(x=xd, **pars)
        ax.plot([-xd, xd], [hd, hd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [aqprops['D'], aqprops['D']], '.', color=clr, label=f"L={pars['L']}")

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    # plt.show()

    # Drawdown
    
    title = f"Blom1D, confined, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}\n" +\
        f"The Extraction Q increases from {Qs[0]} to {Qs[-1]} m2/d" 
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=title, xlabel="x [m]", ylabel="drawdown [m]"),
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q in Qs:
        clr = next(clrs)
        pars['Q'] = Q
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        dd = V1.dd(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        xd = V1.xdiv(**pars)
        ddnd = V1.dd(x=xd, **pars)
        ax.plot([-xd, xd], [ddnd, ddnd], 'v', color=clr, label=f"xd={xd:.3g} m")
        ax.plot([-pars['L'], pars['L']], [0., 0.], '.', color=clr, label=f"L={pars['L']}")

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    # plt.show()

    
    # %% [markdown]
    # ## Voorbeeld Blom 1D, variërende Q
    # 
    # Bij Blom wordt de $L$ berekend, zodanig dat de verlaging op $x=L$ gelijk is aan $N c$.
    # 
    # In dit voorbeeld worden de grondwatertstand en de verlaging berekend voor verschillende waarden van $Q$. Voor elke $Q$ wordt de afstand $L$ berekend waarbinnen de sloten droogvallen. Op deze afstand is de verlaging gelijk aan $N c$, waardoor op afstand $L$ de sloot juist droogvalt (of beter: de sloot daar net niet meer draineert).
    # 
    # We zien verder dat de verlaging in het eerste plaatje, met variabele pakketdikte voor gotere onttrekkingen groter is dan die in het tweede plaatje, voor de situatie met vaste pakketdikte.
    # 
    # Voor de situatie met variabele pakketdikte zou de stijghoogte op $x > L x$ nog iets gecorrigeerd kunnen worden voor de in werkelijkheid afnemende dikte. Dit effect is echter zo klein dat het verschil in de aansluiting op het aangegeven punt, dus $x=L$ in de grafiek niet is te zien. Deze correctie kan eenvoudig worden verwaarloosd in de praktijk.In het onderhavige geval is dit een correctie van de $\lambda van $h/H = (D - s_L) / D \approx 19.5 / 20 \approx 0.98$ op de gebruikte waarde van $\lambda$, dus verwaarloosbaar.

    # %% [markdown]
    ### Blom 1D, fixed L and varying Q
    
    # %%
    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 0.5, 'N': 0.003}
    Qs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    x = np.linspace(0, 1000., 251)[1:]
    L = 250.

    B1 = Blom1D(aqprops=aqprops)
    
    # head
    ax = newfig(f"Blom 1D, head, k={aqprops['k']:.1f} m/d, H={aqprops['D']:.1f} m, Nc={pars['N'] * aqprops['c']:.3g} m, lambda={aqprops['lambda']:.3g} m",
                "x [m]", "drawdown [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for Q in Qs:
        clr = next(clrs)
        pars['Q'] = Q
        L = B1.getL(**pars)
        h = B1.h(x=x, **pars)        
        label=f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = B1.h(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((h[::-1], h)), color=clr, label=label)

        xd = B1.xdiv(**pars)
        hd = B1.h(x=xd, **pars)
        hL = B1.h(x=L, **pars)    
        if not np.isnan(xd):
            ax.plot([-xd, xd], [hd, hd], 'o', color=clr, label=f"xd={xd:.3g} m, hd={hd:.3g} m")
        ax.plot([-L, L], [hL, hL], '.', color=clr, label=f"L={L:.3g} m, hL={hL:.3g} m")

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    # plt.show()

    # Drawdown
    ax = newfig(f"Blom 1D, verlaging, k={aqprops['k']:.1f} m/d, D={aqprops['D']:.1f} m, Nc={pars['N'] * aqprops['c']:.3g} m, lambda={aqprops['lambda']:.3g} m",
                "x [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q in Qs:
        clr = next(clrs)
        pars['Q'] = Q
        L = B1.getL(**pars)  
        label = label=f"Q = {pars['Q']:.3g} m2/d, N={pars['N']:.3g} m/d"
        dd = B1.dd(x=x, **pars)
        ax.plot(np.hstack((-x[::-1], x)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        xd = B1.xdiv(**pars)
        ddnd = B1.dd(x=xd, **pars)    
        ddnL = B1.dd(x=L, **pars)
        if not np.isnan(xd):
            ax.plot([-xd, xd], [ddnd, ddnd], 'o', color=clr, label=f"xd={xd:.3g} m")        
        ax.plot([-L, L], [ddnL, ddnL], '.', color=clr, label=f'L={L:.3g} m, dd={ddnL:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=10, loc='lower right')

    # plt.show()
    
    # %% [markdown]
    ## Voorbeelden Verruijt, axiaal symmetrisch

    # %% [markdown]
    ### Verruijt 1D fixed Q and varying R
    
    # %%
    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 300, 'N': 0.003, 'R': 250.}
    r = np.linspace(0, 300, 301)[1:]

    V2 = wVerruijt(xw=0., yw=0., aqprops=aqprops)

    # Head
    ax = newfig(f"Verruijt axiaal-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for R in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)
        pars['R'] = R
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = V2.h(x=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr, label=label)

        rd  = V2.rdiv(**pars)
        hd = V2.h(x=pars['R'], **pars)
        ax.plot([-rd, rd], [hd, hd], 'v', color=clr, label=f"rd={rd:.3g} m, hd={hd:.3g} m")
        ax.plot([-pars['R'], pars['R']], [aqprops['D'], aqprops['D']], '.', color=clr, label=f"R={pars['R']:.3g} m, h={aqprops['D']:.3g} m")

    ax.text(0.1, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    # plt.show()

    # Drawdown
    ax = newfig(f"Verruijt axial-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for R in [50., 100., 150., 200., 250., 300.]:
        clr = next(clrs)
        pars['R'] = R    
        dd = V2.dd(x=r, **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        rd = V2.rdiv(**pars)
        if not np.isnan(rd):
            ddnd = V2.dd(x=rd, **pars)
            ax.plot([-rd, rd], [ddnd, ddnd], 'v', color=clr, label=f"rd={rd:.3g} m, ddnd={ddnd:.3g} m")
        ax.plot([-R, R], [0., 0.], '.', color=clr, label=f'R={R:.3g} m, ddnd={0.:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    # plt.show()


    # %% [markdown]
    ### Bodemconstanten, gebiedsradius, voeding, onttrekking en putstraal
    
    # %%
    aqprops = {'k': 20, 'D': 20}
    pars = {'Q': 1200., 'N': 0.002, 'R': 1000.}

    V2 = wVerruijt(xw=0., yw=0., aqprops=aqprops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Verruijt k={aqprops['k']:.0f} m/d, H=vast={aqprops['D']:.0f} m, N={pars['N']:.3g} m/d, R={pars['R']:.0f} m")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("h [m]"),
                
    # Kleuren en debieten
    clrs = cycle('brgmck')
    Qs =[0.001, 500, 1000, 2000, 3000]

    r = np.logspace(0, 3, 31)
    R = pars['R']
    H = aqprops['D']

    for Q in Qs:
        pars['Q'] = Q
        clr = next(clrs)    
        # Stijghoogte (links en rechts)
        dd = V2.dd(x=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), H - np.hstack((dd[::-1], dd)), color=clr,
                label=f'Q={Q:.0f} m3/d')

        # Intrekgebied, radius = rI, links en rechts    
        rI = V2.rdiv(**pars)
        if not np.isnan(rI):
            ddnI = V2.dd(x=rI, **pars)
            ax.plot([-rI, rI], [H - ddnI, H - ddnI], ls='none', marker='v', mfc=clr,
                mec='k', ms=6, label=f'r_intr.={rI:.0f} m, dd={ddnI:.3g} m', zorder=5)    

    # Vaste randen
    H = aqprops['D']
    R = pars['R']
    ax.plot([+R, +R], [0, H], '--', color='blue', lw=2, label='vaste rand')
    ax.plot([-R, -R], [0, H], '--', color='blue', lw=2, label='')

    # Put
    rw = 0.5
    ax.plot([+rw, +rw], [0, H], ':', color='blue', lw=2, label='put')
    ax.plot([-rw, -rw], [0, H], ':', color='blue', lw=2, label='')

    # Pakket bodem
    ax.add_patch(patches.Rectangle((-R, -2), 2 * R, 2, fc='gray'))

    ax.text(0.6, 0.6, "Variabele pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    leg = ax.legend(loc='lower left', fontsize=10)
    leg.set_bbox_to_anchor((0.10, 0.11, 0.3, 0.5), transform=ax.transAxes)

    # %% [markdown]
    ### Bodemconstanten, gebiedsradius, voeding, onttrekking en putstraal
    
    # %%
    aqprops = {'k': 10, 'D': 20}
    pars = {'Q': 1200., 'N': 0.002, 'R':1000.}

    V2 = wVerruijt(xw=0., yw=0., aqprops=aqprops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Verruijt k={aqprops['k']:.0f} m/d, D=variabel, HR={aqprops['D']:.0f} m, N={pars['N']:.3g} m/d, R={pars['R']:.0f} m")
    ax.set_xlabel("r [m]")
    ax.set_ylabel("h [m]"),
                
    # Kleuren en debieten
    clrs = cycle('brgmck')
    Qs =[0.001, 500, 1000, 1500, 2000] #  2500]

    r = np.logspace(0, 3, 31)
    R = pars['R']

    for Q in Qs:
        pars['Q'] = Q
        clr = next(clrs)    
        # Stijghoogte (links en rechts)
        h = V2.h(x=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr,
                label=f'Q = {Q:.0f}')

        # Intrekgebied, radius = rI, links en rechts
        rI = np.sqrt(Q / (np.pi * pars['N']))
        rI = V2.rdiv(**pars)
        if not np.isnan(rI):
            hI = V2.h(x=rI, **pars)
            ax.plot([-rI, rI], [hI, hI], ls='none', marker='v', mfc=clr,
                mec='k', ms=6, label='intrekgrens', zorder=5)    

    # Vaste randen
    H = aqprops['D']
    ax.plot([+R, +R], [0, H], '--', color='blue', lw=2, label='vaste rand')
    ax.plot([-R, -R], [0, H], '--', color='blue', lw=2, label='')

    # Put
    rw = 0.5
    ax.plot([+rw, +rw], [0, H], ':', color='blue', lw=2, label='put')
    ax.plot([-rw, -rw], [0, H], ':', color='blue', lw=2, label='')

    # Pakket bodem
    ax.add_patch(patches.Rectangle((-R, -2), 2 * R, 2, fc='gray'))

    ax.text(0.6, 0.6, "Vaste pakketdikte", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    leg = ax.legend(loc='lower left', fontsize=10)
    leg.set_bbox_to_anchor((0.10, 0.11, 0.3, 0.5), transform=ax.transAxes)
    
    # plt.show()

    # %% [markdown]
    ## Voorbeeld Blom, axiaal symmetrisch

    # %% [markdown]
    ### Blom axiaal symmetrisch, fixed Q and varying R

    # %%
    aqprops = {'k': 10., 'D': 20., 'c': 200.}
    pars = {'Q': 300, 'N': 0.003}
    r = np.linspace(0, 300, 301)[1:]

    B2 = wBlom(xw=0., yw=0., aqprops=aqprops)

    # Head
    ax = newfig(f"Blom, axiaal-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "head [m]",
                figsize=(10, 6))

    clrs = cycle('rbgmkcy')
    for Q in np.array([1., 1.5, 2., 2.5, 3.0]) * 500.:
        clr = next(clrs)
        pars['Q'] = Q
        R = B2.getR(R=r.mean(), **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        h = B2.h(x=r, **pars)
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((h[::-1], h)), color=clr, label=label)

        rd = B2.rdivh(**pars)
        hd = B2.h(x=rd, **pars)
        ax.plot([-rd, rd], [hd, hd], 'o', color=clr, label=f"rd={rd:.3g} m, hd={hd:.3g} m")
        HR = B2.aq['D'] - pars['N'] * B2.aq['c']
        ax.plot([-R, R], [HR, HR], '.', color=clr, label=f'R={R:.3g} m, h={HR:.3g} m')

    ax.text(0.1, 0.6, "Variabele pakketdikte voor r < R", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    # plt.show()

    # %%
    # Drawdown
    B2 = wBlom(xw=0., yw=0., aqprops=aqprops)

    ax = newfig(f"Blom, axial-symmetrisch, k={aqprops['k']:.1f}, D={aqprops['D']:.1f}",
                "r [m]", "drawdown [m]",
                figsize=(10, 6))
    ax.invert_yaxis()

    clrs = cycle('rbgmkcy')
    for Q in np.array([1., 1.5, 2., 2.5, 3.0]) * 500.:
        clr = next(clrs)
        pars['Q'] = Q
        R = B2.getR(R=r.mean(), **pars)    
        dd = B2.dd(x=r, **pars)
        label = f"Q = {pars['Q']:.3g}, N={pars['N']:.3g}"
        ax.plot(np.hstack((-r[::-1], r)), np.hstack((dd[::-1], dd)), color=clr, label=label)

        ddR = B2.dd(x=R, **pars)
        ax.plot([-R, R], [ddR, ddR], '.', color=clr, label=f'R={R:.3g} m, ddnd={0.:.3g} m')

    ax.text(0.1, 0.6, "Vaste pakketdikte voor r < R", transform=ax.transAxes, bbox=dict(facecolor='gray', alpha=0.3))
    ax.legend(fontsize=6, loc='lower right')

    # plt.show()

    # %% [markdown]
    ### Demonstratie van de voortgang van het iteratieproces volgens Newton om R te vinden waar de verlaging gelijk is aan Nc
    # 
    # De afstand van de put waarop de verlaging precies gelijk is aan Nc, het critierium voor juist droogvallen van de sloten wordt iteratief berekend met de methode van Newton. Het voorschrijden van dit iteatieproces wordt hieronder grafisch weergegeven.
    # 
    # Voor de iteraties is de afgeleide nodig van de fuctie $y(R)$ zie boven. De tweede grafiek toont de afgeleide, zowel analytisch als numeriek berekent ter controle.

    # %%
    aqprops = {'k': 30.0, 'D': 20.0, 'c': 200.0}
    pars = {'Q': 1200., 'N':0.02, 'R': 1.0}

    B2 = wBlom(xw=0., yw=0., aqprops=aqprops)
    ax = B2.plot_newton_progress(R=1., Q=1500.0, N=0.002)
    ax.set_ylim(-0.5, 2.5)

    ax = B2.plot_derivative_of_y(R=1, Q=1200.0, N=0.002)

    # plt.show()

    # %% Bruggeman 370_01 example
    # Drawdown due to a well in a leaky aquifer were kD and c jump at x=0 
    
    # Aquifer parameters
    kD1, kD2, c1, c2 = 250., 1000., 250, 1000.
    aqprops = {'kD1': kD1, 'kD2': kD2, 'c1': c1, 'c2': c2}
    Q = 1500. # m3/d
    pars = {'Q': Q}
    # Coordinates
    x = np.linspace(-2000, 2000, 401)
    y = None

    # Well position
    xw = 200.

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.set_title("Bruggeman 370_01, verification, \n" +
                 fr"xw={xw:.4g} m, Qw={Q:.4g} m3/d\nkD1={kD1:.4g} m2/d, kD2={kD2:.4g} m2/d, c1={c1:.4g} d, c2={c2:.4g} d", fontsize=12)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("phi [m]")
    ax.grid(True)


    # Show symmetrie for xw > 0 and xw < 0 by interchanging kD1 <-> kD2, c1 <-> c2 
    Br1 = Brug370_1(xw=xw, yw=0., aqprops={'kD1': kD1, 'kD2': kD2, 'c1': c1, 'c2': c2})
    X, Y, Phi1 = Br1.dd(x, y, **pars)

    Br2 = Brug370_1(xw=xw, yw=0., aqprops={'kD1': kD2, 'kD2': kD1, 'c1': c2, 'c2': c1})
    X, Y, Phi2 = Br2.dd(x, y, **pars)

    # Compare Brug370_01 with De Glee leaky aquifer solution
    Br3 = Brug370_1(xw=xw, yw=0, aqprops={'kD1': kD1, 'kD2': kD1, 'c1': c1, 'c2': c1})
    X, Y, Phi3 = Br3.dd(x, y, **pars)
    
    # Using same kD2 and c2 should give De Glee
    dGlee1 = wDeGlee(xw=xw, yw=0, aqprops={'kD': kD1, 'c': c1})
    PhiGl1 = dGlee1.dd(x, y, **pars)

    # Compare Brug370_01 with De Glee leaky aquifer solution
    Br4 = Brug370_1(xw=xw, yw=0, aqprops={'kD1': kD2, 'kD2': kD2, 'c1': c2, 'c2': c2})
    X, Y, Phi4 = Br4.dd(x, y, **pars)

    # Using same kD1 and c1 should give De Glee
    dGlee2 = wDeGlee(xw=xw, yw=0, aqprops={'kD': kD2, 'c': c2})
    PhiGl2 = dGlee1.dd(x, y, **pars)

    # Show the combined results
    if X.ndim == 1:
        ax.invert_yaxis()
        ax.plot(X, Phi1, label=f"Brug initial   {Br1.aq}")
        ax.plot(X, Phi2, label=f"Brug reversed, {Br2.aq}")
        ax.plot(X, Phi3, label=f"As dGlee1, {Br3.aq}")
        ax.plot(X, Phi4, '.', label=f"As dGlee2, {Br4.aq}")
        ax.plot(X, PhiGl1, label=f"dGlee1, {dGlee1.aq}")
        ax.plot(X, PhiGl2, '.', label=f"dGlee2, {dGlee2.aq}")
    else: # If X and Y are 2D, then contour
        levels = 15
        CS = ax.contour(X, Y, Phi1, levels=levels)
        ax.clabel(CS, levels=CS.levels, fmt="{:.2f}")
        
    ax.legend(loc='best', fontsize='x-small')
    
    
    plt.show()


# %%
