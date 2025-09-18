# %% [markdown]
# Kinematic wave of moisture through unsaturated zone
#
# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation  # , FFMpegWriter
from scipy.integrate import simpson
import etc

import warnings
warnings.filterwarnings("error")

cwd = os.getcwd()
dirs = etc.Dirs(cwd)
print("Project directory (cwd): `{wd}` ")
    
dirs.add_dir('videos')

# --- Local modules
sys.path.insert(0, "")

from src.rootz_rch_model  import RchEarth # noqa
from src.NL_soils import Soil #noqa

# --- update figure settings
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 5
})

# %%
class Kinematic_wave():
    """Class to represent the moisture profile using kinematic waves.
    
    The kinematic wave model is used to simulate the movement of moisture through the unsaturated zone.
    The profile consists of points that represent the soil moisture content at different depths over time. Each point carries its time, depth, velocity and the upstream and downstream soil moisture content (theta1 and theta2). When a point is a moisture front, the downstream soil moisture content is lower than the upstream soil moisture content.
    The velocity of the wave is calculated based on the soil moisture content and the saturated hydraulic conductivity.
    The soil moisture content is always between theta_r (residual) and theta_s (saturated).
    The initial condition for the soil moisture content can be set using different scenarios.
    The class provides methods to update the profile over time, calculate the velocity of the wave, and plot the profile or all profiles at once showing the development of the moisture front over time.

    @ TO 2025-08-01
    """
    
    # --- dtype of kinematic wave profile
    profile_dtype = np.dtype([
                        ('t', '<f8'),      # Time (d)
                        ('tst1', '<f8'),   # start time of point (upstream side)
                        ('tst2', '<f8'),   # start time of point (downstream side)
                        ('z', '<f8'),      # Depth (m)
                        ('v', '<f8'),      # Velocity (m/d)
                        ('theta1', '<f8'), # Upstream soil moisture content (m^3/m^3)
                        ('theta2', '<f8'), # Downstream soil moisture content (m^3/m^3),
                        ('front', '?')     # indicats the point is a sharp front
                        ])

    # --- dtype of input data, i.e. of initial profile
    z_theta_dtype = np.dtype([('z', '<f8'), ('theta', '<f8')])
    
    
    def __init__(self, soil: Soil, z_theta: np.ndarray =None)-> None:
        """Initialize the kinematic wave profile using props in soil.
        
        Parameters
        ----------
        soil: soil.snl.Soil object
            A dutch soil with its methods and properties
        theta: float | ndarray
            Initial moisture concent, either one value or an array
        z_theta: np.ndarray of np.dtype([('z', float), ('theta', float)])
            Depth of water table below root zone in cm.
        """        
        self.soil = soil
        self.profile = self.set_profile(z_theta) # Also sets self.z0
        
         # --- Intermediate profiles for later animation
        self.profiles={}
        return None
        
    def set_profile(self, z_theta: np.ndarray)->np.ndarray:
        """Return the initial moisture profile.
        
        A profile consists represents the moisture situation between the bottom
        of teh root zone and the water table (or the end of the percolation czone).
        
        A profile is a recarray with records ['z', 'v', 'theta1', 'theta2']
        Each record represents a (moisture) point moving along with constant speed (v)
        that is determined by the soil and the local moisture content. The velocity
        if th velocity with which moisture profiel points move. If the v were constant
        along the profile, it would be the speed at which the entire moisture profile 
        wanders downward. However points move with speed determined by their moisture
        content and, therfore, different points move at different speed, and points
        representing higher moisture content overtake downstream points with lowe
        moisture content, thereby creating sharp moisture fronts.
        
        Each point has an upstream and a downstream point. If they are not the same
        the downstream point is the point with lowest theta and the point then
        represents a sharp front, which originates from having been overtaken by downstream points.
        
        The speed of a (moisture profile) point is uniquely determined by the combination
        of its upstream and downstream moisture content and the relation K(theta), or
        rather dK(theta)/dtheta pertaining to the soil.
        
        Parameters
        ----------
        z_theta: np.ndarray with dtype([('z', float), ('theta', float)])
            Initial moisture profile. Can of arbitrary length.
            The first z determines the depth of the root zone, which is
            the start of the profile. The second z determines the depth
            of the profile, i.e. the elevation of the groundwater table.
            
            Values of z must be in cm, coording to dimensions in soil.
            To initialyze with field capacity prepare z_theta using
            theta = soil.theta_fc()
            
            The wave or point velocities will be computed from soil and
            theta as dK_dtheta(theta).
        """
        prof = np.zeros(len(z_theta), dtype=self.__class__.profile_dtype)
        
        # --- Verify input of initial profile
        if not isinstance(z_theta, np.ndarray) \
            or z_theta.dtype != self.__class__.z_theta_dtype:
            raise TypeError(f"z_theta must be an np.ndarray of dtype: {self.__class__.z_theta_dtype}")
        
        # --- Set depth z of bottom of root zone
        self.z0 = z_theta['z'][0]
        
        # --- Start with t, tst1 and tst2 all zero        
        prof['t'] = 0.
        prof['z'] = z_theta['z']
        prof['theta1'] = self.soil.theta_fc()
        prof['theta2'] = self.soil.theta_fc()        
        prof['v'] = self.soil.dK_dtheta(self.soil.theta_fc())
        
        # --- Set tst1 and tst2 such that computed theta will equal theta_fc
        prof['tst1'] = -(prof['z'] - self.z0) / prof['v']
        prof['tst2'] = -(prof['z'] - self.z0) / prof['v']
        
        return prof

    
    def point_velocities(self, th1: float | np.ndarray, th2: float | np.ndarray)-> float | np.ndarray:
        """Return the velocity of all the wave points velocities.

        Parameters
        ----------
        th1 : float
            Upstream soil moisture content 1 (m^3/m^3).
        th2 : float
            Downstream soil moisture content 2 (m^3/m^3).
        epsilon : float, optional
            Pore size distribution parameter, by default 3.5.
        Returns
        -------
        float
            Velocity of the wave profile (m/d).
        """        
        theta_s = soil.theta_s
        theta_r = soil.theta_r
        
        th1, th2 = np.atleast_1d(th1, th2)  # Ensure inputs are arrays
        
        th1 = np.fmax(th1, theta_r)
        th2 = np.fmax(th2, theta_r)
        
        th1 = np.fmin(th1, theta_s)
        th2 = np.fmin(th2, theta_s)
        
        v = np.zeros_like(th1)

        # Velocity of points upstream of sharp front
        vth1 = np.atleast_1d(self.soil.dK_dtheta(th1))
        vth2 = np.atleast_1d(self.soil.dK_dtheta(th2))

        # --- Bool array separating sharp front points from ordinary points
        L = np.isclose(th1, th2)
        
        # --- Velocity of normal points (no sharp fronts)
        if np.any(L):
            v[L] = vth1[L]

        # --- Veolocity of sharp fronts
        if np.any(~L):
            v[~L] = (self.soil.K_fr_theta(th1[~L]) - self.soil.K_fr_theta(th2[~L])
                 ) / (th1[~L] - th2[~L])
    
        return (vth1.item() if vth1.size == 1 else vth1,
                v.item()    if v.size    == 1 else v,
                vth2.item() if vth2.size ==1 else  vth2)


    def get_shock_times(self, vtol=1e-6):
        """Return shock times, i.e. times when points overtake the next point.

        Negative values imply that no shock can ever occur, the successive points diverge over time.
        Positive times indicate that shocks will occur at these times in the future
        if conditions remain the same.
        
        Parameters
        ----------
        vtol : float, optional, default 1e-6
            prevents division by zero if velocities are the same, which happens
            when generating several new points at once at the same depth every
            time when the recharge q is less than its previous value.
        """
        prof = self.profile
        t = prof[0]['t']
        
        # --- Avoid division by zero for points that happen to have the same z at given time
        dv = np.fmax(vtol, prof['v'][:-1] - prof['v'][1:])
        
        # --- Compute shock time for all points of profile        
        tshock = t + (prof['z'][1:] - prof['z'][:-1]) / dv
        tshock[tshock <= t] = np.inf
        
        return tshock
    
    def front_step(self, t_next, tol=1e-8):
        """Move the poinnts in the profile to t_next (predictor - corrector method).
        
        Does not correct for a shock.
        
        For every sharp front, and there may be many simultaneously within the
        profile, theta changes continuously, and so does its velocity.
        We, therefore, seek to move points with velocity that is corrected
        for this continueously changing velocity.
        
        This predictor-corrector scheme is expected to increase the accuracy.
        (But I don't know if it's really required).                
        """
        # --- first estimate of the new point positions and velocities

        dt = t_next - self.profile['t'][0]
        # z = self.profile['z'] + self.profile['v'] * dt
        # x = z - self.z0
        # t_travelled1 = t_next - self.profile['tst1']
        # t_travelled2 = t_next - self.profile['tst2']
        # vavg1 = x / np.fmax(tol, t_travelled1)
        # vavg2 = x / np.fmax(tol, t_travelled2)

        # --- estimates of new theta
        # self.profile['theta1'] = self.soil.theta_fr_V(vavg1)
        # self.profile['theta2'] = self.soil.theta_fr_V(vavg2)

        # v  = self.point_velocities(self.profile['theta1'], self.profile['theta2'])[1]
        # self.profile['v'] = v

        # --- update profile with correcte values
        self.profile['z'] += self.profile['v'] * dt
        self.profile['t']  = t_next
        
        # --- return updated profile
        return None


    def update(self, dt):
        """Update the profile based on the time step.
        
        All points move with their own velocity. However, some
        points may overtake other points within the time step.
        So we first compute the shock times, move the points
        over the minimum shocktimes, update the velocity of the
        shockpoint while deleting the point that was overtaken
        and repeat the procedure until all points have been moved
        over the time step dt        
        """
        t = self.profile['t'][0]
        t1 = t + dt # to start time, t1 end time, dt apart

        while True:
            tsh = self.get_shock_times() # tsh is the time when the shock occurs
            
            # --- Find the indix of the point with the lowest shock time ---
            # --- This can be any point, not just the first one!
            ip_min = np.argmin(tsh)

            if tsh[ip_min] <= t1:  # There are shocks in this time step           

                # --- Find the first shock time between t0 and t1
                tshock = tsh[ip_min]
                
                # --- Update velocity of this shock point
                # --- Also updates t to tshock
                self.front_step(tshock)
                                
                # --- Remove the point that was overtaken
                self.profile[ip_min]['theta2'] = self.profile[ip_min + 1]['theta2']
                self.profile[ip_min]['tst2']   = self.profile[ip_min + 1]['tst2']
                self.profile[ip_min]['v']      = self.point_velocities(self.profile[ip_min]['theta1'], self.profile[ip_min]['theta2'])[1]
                self.profile[ip_min]['front']  = True
                self.profile = np.delete(self.profile, ip_min + 1)
            else:
                # --- No shocks in (remaining part of) this time step
                # --- just update the profile to t1
                self.front_step(t1)
                
                # --- adapt front theta                                
                # TODO -- continuous correction of theta for front points
                # dtheta/dt = -(VL - vF) partial (theta /partial z)                
                # Ifr = np.where(self.profile['front'])[0]                
    
                # if len(Ifr) > 0:
                    # Ifr = Ifr[Ifr > 0]
                    # if len(Ifr) > 0:
                        # vth1, v, _ = self.point_velocities(self.profile['theta1'][Ifr], self.profile['theta2'][Ifr])                
                        # dth_dz = (self.profile['theta1'][Ifr] - self.profile['theta2'][Ifr - 1]) / (self.profile['z'][Ifr] - self.profile['z'][Ifr - 1])
                        # self.profile['theta1'][Ifr] -= (vth1 - v) * dt * dth_dz
                
                # --- leave the while loop
                break
                
        return None
    

    def prepend(self, t, q, N=15, tol = 0.001):
        """Generate a new point and prepend it to the profile.
        
        If the q of this time step is less than that of the previoous time step
        then two points are prepended, one with the previous theta (finishing that
        time step with its q) and one with the new and lower theta, starting the
        new time step with the lower q. This way these two points can and will
        wander with in ever growing gap between them, representing the tail of
        the previous time step.
        If the q of this time step is larger than that of the previous time step,
        then one new point is generated with the higher theta at it's upstream side
        and lower theta at it's downstream side, thus reprsenting a shock.
        
        Parameters
        ----------
        t: scalar [d]
            current time at which a point is inserted
        q: scalar [cm/d]
            infiltration from root zone for the current time step
        tol: scalar
            theta tolerance to check if q larger or smaller than previous
        """
        # --- previous point
        previous = self.profile[0]
                        
        # --- the first point must have been updated to t > 0
        # --- else delete the first point, as it should not be here
        if previous['t'] == 0:
            self.profile = self.profile[1:]
        
        # --- get theta pertaining to current q (K(theta) = q)
        theta1 = max(soil.theta_fc(), soil.theta_fr_K(q))
        theta2 = previous['theta1']    
        
        # --- a front is when theta of current point > theta previous point
        front  = theta1 > theta2 + tol
        same_q = np.isclose(theta1, theta2, tol)

        if same_q:
            # --- normal points when q0_prev == q ---
            current = np.zeros(1, self.__class__.profile_dtype)
            current['theta1'] = theta1
            current['theta2'] = theta1
            current['tst1'] = t
            current['tst2'] = t
        elif front:   # tol in cm tol =0.005 cm = 0.05 mm
            # --- shock, sharp front ----
            current = np.zeros(1, self.__class__.profile_dtype)
            current['theta1'] = theta1
            current['theta2'] = theta2
            current['tst1'] = t
            current['tst2'] = previous['tst1']
        else:
            # --- tail, new points are generated at the same time and z ---
            N = max(1, int((theta2 - theta1) * 1000))
            current = np.zeros(N, self.__class__.profile_dtype)
            thetas = np.linspace(theta1, theta2, N)
            current['theta1'] = thetas
            current['theta2'] = thetas
            current['tst1']   = t
            current['tst2']   = t
            
        current['z']  = self.z0
        current['t']  = t
        current['v'] = self.point_velocities(current['theta1'], current['theta2'])[1]
        
        # --- prepend h
        self.profile =np.concatenate([current, self.profile])

        return None
        
    
    def q_at_z(self, z_gwt, dz=10):
        """Return the downward flux at z at current time.
        
        Can be used to get the flux through the water table at z_gwt.
        Works even with z_gwt varying over time.
        
        """        
        p = self.profile

        # --- Points between which the z_gwt intersects
        i1 = np.where(p['z'] < z_gwt)[0][-1]
        i2 = np.where(p['z'] >=  z_gwt)[ 0][0]
        p1, p2 = p[i1], p[i2]

        if np.isclose(p1['theta2'], p2['theta1'], rtol=1e-3):
            theta_gwt = p1['theta2']            
        else: # Interpolate the moisture content at z_gwt       
            z1, z2 = p1['z'], p2['z'] 
            N = int(2 + (z1  -z1) / dz)
            z = np.linspace(z1, z2, N)
            v_avg = (z - self.z0) / (p1['t'] - p1['tst2'])
            theta = self.soil.theta_fr_V(v_avg)
            theta_gwt = np.interp(z_gwt, z, theta)

        # --- Flux at z_Gwt = K(theta_gwt)
        q_gwt = self.soil.K_fr_theta(theta_gwt)
        return q_gwt

    
    def get_profiles_obj(self, t: float, date: np.datetime64)-> dict:
        """Return dict that is added to the profiles for later plotting."""
        
        # --- get the interpolated z, theta of the profile
        z, theta = self.get_profile_line()
        
        iz = np.where(z <= self.z_gwt)[0][-1]
        
        theta_iz = np.interp(self.z_gwt, z[iz:], theta[iz:])
        
        z     = np.hstack((z[:iz], self.z_gwt))
        theta = np.hstack((theta[:iz], theta_iz))
        
        # --- Compute how amount of moisture is in the profile above fc
        V = my_simpson(theta - self.soil.theta_fc(), x=z)
        
        return {'profile': self.profile.copy(),
                    't':t,
                    'date': date,
                    'line': (z, theta),
                    'V': V}
        
    def get_Vol(self):
        """Return moisture volume in profile over time.
        """
        dtype=np.dtype([('t', '<f8'),('date', 'datetime64[ns]'),('V', '<f8')])
        
        vol = np.zeros(len(self.profiles), dtype=dtype)
        
        for ip, prof in self.profiles.items():
            vol[ip] = (prof['t'], prof['date'], prof['V'])
        
        return vol

        
    def simulate(self, recharge):
        """Simulate the percolation given the recharge from the rootzone.
        
        In fact, a simulation is handling and changing the profile object.
        You dont' have to keep it, but you can issue the graph as a
        graph from it.
        
        Parameters
        ----------
        recharge: pd.DataFrame with field 'RCH'
            DataFrame with recharge time series.                
        """
        mm_per_cm = 10 # for converting cm to mm and vice versa
        
        # --- initialize or set column with recharge at the groundwater table
        recharge['qwt'] = 0.        
        
        # --- get time in days as floats from datetime index        
        time = (recharge.index - recharge.index[0]) / np.timedelta64(1, 'D')
        
        dt = np.diff(time)[0]
        
        # --- Convert pd.DataFrame to array with dtype of fields
        recharge = recharge.to_records()
        
        # --- Simulate timestep by timestep, record by record
        # --- The very first record has z=self.z0, t=0, tst1=0, tst2=0, than theta_fc, theta_fc
        for ip, (t, pe) in enumerate(zip(time, recharge)):
        
            if np.mod(ip, 10) == 0:
                pass

            # --- next flux from root zone
            # --- (must be in cm/d)
            q0 = pe['RCH'] / mm_per_cm
                
            # --- Generate and prepend new record(s) to profile            
            self.prepend(t, q0)

            # --- Store current profile for later animation ---
            # --- Store profiel at time = t (at the beginning of the day)
            self.profiles[ip]=self.get_profiles_obj(t, date=pe['index'])

            # --- Update the profile to end of the day (t1 = t + dt)
            # --- evaluate shocktimes
            # --- move points over dt
            self.update(dt)
                    
            # --- Get flux at water table
            qwt = self.q_at_z(self.z_gwt)
            pe['qwt'] = qwt * mm_per_cm # cm/d --> mm/d (store as mm/d)

            # --- clip points below z_gwt
            z = self.profile['z']
            Ip = np.arange(len(z))[z > z_gwt]
            if len(Ip) > 1:
                self.profile = self.profile[:Ip[1]]
        
        # --- Store profile at end of the last day
        # --- Get last time step size
        dt = np.diff(time)[-1]
        self.profiles[ip]=self.get_profiles_obj(t + dt, date=pe['index'] + np.timedelta64(1, 'D'))
                            
        return pd.DataFrame(recharge)
      

    def get_profile_line(self, dz=10):
        """Interpolate between profiles to get a continuous smooth wave between the sub profiles.
        
        We have a single large profile over the total depth of the percolation zone.
        Subprofiles are parts of the overall profile between sharp fronts. These
        subprofiles may be long but with only few points to actually define them.
        Therefore, to keep smooth subprofiles, we use this function to itnerpolate
        points between them. Perhaps the profile could be inspected for parts to
        gain more points.
                
        """
        # --- line points       
        z, theta = [], []
        
        # --- consecutive profile records     
        for (p1, p2) in zip(self.profile[:-1], self.profile[1:]):
            z1, z2 = p1['z'], p2['z']
            t = p1['t'] # all records have he same t
            
            # --- number of points to be interpolated (>= dz cm apart)
            N = 2 + int((z2 - z1) / dz)

            # --- If the two thetas are the same, constant theta, no interpolation necessary
            if np.isclose(p1['theta2'], p2['theta1']) or N < 3:
                z     += [z1, z2]
                theta += [p1['theta2'], p2['theta1']]
                continue
            else:
                # --- interpolate N points including ends                
                z_ = np.linspace(z1, z2, N)
                v_avg  = (z_ - self.z0) / (t - p2['tst1'])
                theta_ = self.soil.theta_fr_V(v_avg)
                z     += list(z_)
                theta += list(theta_)
        return np.array(z), np.array(theta)

    def plot_profiles(self, ax=None):
        """Plot the profiles on given axes."""
        if ax is None:
            ax = etc.newfig("Profile", "z [cm]", "theta", figsize=(10, 6))

        for ip in self.profiles:
            z, theta = self.profiles[ip]['line']
            t = self.profiles[ip]['t']
            ax.plot(z, theta, label=f"t = {t:.3g} d")


def make_animation(profiles: dict, soil: Soil, z_gwt: float)->tuple:

    # --- determine extent of axes ---    
    z = profiles[0]['line'][0]
    zmin = z[0]
    zmax = z[-1]
    zmax = z_gwt
        
    theta_min = soil.theta_r
    theta_max = soil.theta_s
        
    fig, ax = plt.subplots()
    
    # --- axes labels ---
    title = (f"Kinematic wave for soil {soil.code} {soil.props['Omschrijving']}\n"
        + fr"$K_s$={soil.props['Ks']:.3g} cm/d, "
        + fr"$\theta_{{fc}}$={soil.theta_fc():.3f}, "
        + fr"$\theta_{{wp}}$={soil.theta_wp():.3f}")
    ax.set_title(title)
    ax.set(xlabel='z [cm]', ylabel='theta' )

    # --- extent of plot ---
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(theta_min, theta_max)

    # --- static reference lines (drawn once, stay forever) ---    
    ax.axhline(y=soil.theta_fc(),      color='g', label='field capacity')
    ax.axhline(y=soil.theta_fr_K(0.1), color='m', label='theta q=1 mm/d')
    ax.axhline(y=soil.theta_fr_K(0.2), color='r', label='theta q=2 mm/d')
    
    # --- animated artists (updated ech frame) ---
    line, = ax.plot([], [], lw=2)
    txt = ax.text(0.4, 0.5, f"t = {0.:.0f} d",
                   transform=ax.transAxes,
                   ha='center', va='center',
                   bbox=dict(
                        facecolor="gold",   # background color
                        edgecolor="black",  # border color
                        boxstyle="square"
                   )
                   )
    ax.legend()
    
    # --- Init_func: set up the initial state
    def init_func():        
        line.set_data([], [])
        txt.set_text('')        
        return (line, txt)
    
    # --- Update_func: uses the closure to access `line` and `profiles`
    def update_func(frame):
        p = profiles[frame]
        (z, theta), date = p['line'], p['date']       
        line.set_xdata(z)
        line.set_ydata(theta)
        txt.set_text(f"{np.datetime64(date).astype('datetime64[D]')}, t={frame} d")
        # --- show progress
        if np.mod(frame + 1, 100) == 0:
            print('.', end="")
        if np.mod(frame + 1, 1000) == 0:
            print(frame)
        return (line, txt)
    
    return fig, init_func, update_func

def my_simpson(y, x=None):
    dx = np.diff(x)
    ym = 0.5 * (y[:-1] + y[1:])
    return np.sum(dx * ym)

def test_accuracy(self):
    """Show the accuracy of K(theta) and theta(K(theta)), dK/dtheta and theta(dK(theta)/dtheta).
    
    For any soil:
        * theta(K(theta))         should return theta
        * theta(dK(theta)/dtheta) should return theta
    """
    Soil.load_soils(os.path.join(dirs.data, "NL_VG_soilprops.xlsx"))
    
    codes = ['B01', 'O01', 'B05', 'O05']
    for code in codes:
        sl = Soil(code)
        print(f"{code} fc={sl.theta_fc():.4f}, wp = {sl.theta_wp():.4f}")
        
        for theta in [sl.theta_fc(), sl.theta_wp()]:
            print("{} theta: {:6.4f} ->K(theta): {:8.4g} -> theta(K(theta)): {:6.4f}".format(
                code,
                theta,
                sl.K_fr_theta(theta),
                sl.theta_fr_K(sl.K_fr_theta(theta))))
            
            print("{} theta: {:6.4f} ->V=dK(theta)/dtheta: {:8.4g} -> theta_fr_V(dK(theta)/dtheta): {:6.4f}".format(
                code,
                theta,
                sl.dK_dtheta(theta),
                sl.theta_fr_V(sl.dK_dtheta(theta))))


def change_meteo_for_testing(meteo: pd.DataFrame, N: float=180, m:float = 1)-> pd.DataFrame:
    """Return meteo with adapted 'RCH' for easier testing and checking."""
    
    qrz = meteo['RCH'].values
    qrz = np.zeros(len(meteo), dtype=float)
    
    # --- Make m N-day blocks of continuous q alternating between 0 and 1 mm/d ---
    for i in np.arange(0, 2 * N * m, 2 * N):        
        qrz[i + N:i + 2 * N] = 2. # mm/d
        
    meteo.loc[:, 'RCH'] = qrz
    return meteo
    
# %%

if __name__ == "__main__":
    
    # %% --- Setup the example usage
    
    # Simulate the flow through the unsaturated zone using the kinematic wave model.
    # The input is a time series of recharge on a daily basis.
    # When this works a root-zone module will be inserted to simulate the storage of water in the root zone.
    
    # --- Get weather data for De Bilt
    meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
    os.path.isfile(meteo_csv)
    deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)
    
    # --- Shorten the series for convenience
    deBilt_short = deBilt.loc[deBilt.index >= np.datetime64("2020-01-01"), :]
        
    # --- Set storage capacities of Interception and Rootzone reservoirs
    Smax_I, Smax_R = 1.5, 100 # mm
    
    date_span = (np.datetime64("2020-01-01"), np.datetime64("2021-03-31"))
    
    # --- Get the recharge simulator
    rch_simulator = RchEarth(Smax_I=Smax_I, Smax_R=Smax_R, lam=None)
    
    # --- Compute the recharge
    rch = rch_simulator.simulate(deBilt_short)
    
    # --- If True, replace rch DataFrame field ['RCH'] with a testing pattern
    if True:
        rch = change_meteo_for_testing(rch, N=10, m=1)[rch.index < rch.index[0] + np.timedelta64(100, 'D')]
        # rch.loc[rch.index > rch.index[0] + np.timedelta64(10, 'D'), 'RCH'] = 2

    # %% --- Second step get the soil and simulate the kinematic wave
    Soil.load_soils(os.path.join(dirs.data, "NL_VG_soilprops.xlsx"))
    
    # --- Choose a soil from the Staringreeks
    soil = Soil('O05')
    
    # %% --- Simulate and animate the kinematic wave over time
    
    # --- Initialize the Kinematic Wave profile
    
    # --- Root zone and gwroundwater table depth in cm
    z_rz, z_gwt = 0, 150.
    
    # --- Generate the initial profile
    z_theta = np.zeros(10, dtype=Kinematic_wave.z_theta_dtype)
    z_theta['z'] = np.linspace(z_rz, z_gwt, 10)
    z_theta['theta'] = soil.theta_fc()

    # --- Initiate the kinematic wave object
    kwave = Kinematic_wave(soil=soil, z_theta=z_theta) # z_GWT ?
    kwave.z_gwt = z_gwt
    
    # --- Simulate the kinematic wave
    rch_gwt = kwave.simulate(rch)
    
    # --- Setup of the animation    
    fig, init_func, update_func = make_animation(kwave.profiles, soil, kwave.z_gwt)
    
    # --- Animate
    print(f"Running animation, showing progress, one dot per 100 frames, total number of frames: {len(rch_gwt)}")
    ani = FuncAnimation(fig, update_func, frames=len(kwave.profiles), init_func=init_func,
                        blit=True, repeat=False)
    
    # --- Save anaimation
    ani.save(f"Kinematic_wave_soil {soil.code}.mp4", writer="ffmpeg", fps=20)

    plt.close(fig)
    
    print("Done animation.")
    
    # --- plot RCH and qwt
    ax = etc.newfig(f"q at the water table, z0={z_rz} cm, z_gwt={z_gwt} cm", "time", "q mm/d")
    ax.plot(rch_gwt.index, rch_gwt['RCH'], label='qrtz')
    ax.plot(rch_gwt.index, rch_gwt['qwt'], label='qwt')
    ax.grid(True)
    ax.legend()
    ax.figure.savefig(f"qwt_{soil.code}")

    # --- plot integrated curve of RCH and qwt
    ax = etc.newfig(f"Flux q integrated over time from rootzone and at the water table, z_rz={z_rz}, z_gwt={z_gwt} cm",
                    "time", "integral(q) mm")
    ax.plot(rch_gwt.index, rch_gwt['RCH'].cumsum(), label='qrtz')
    ax.plot(rch_gwt.index, rch_gwt['qwt'].cumsum(), label='qwt')
    ax.grid(True)
    ax.legend()
    ax.figure.savefig(f"qwt_{soil.code}_cumsum")

    # --- plot the volume under the profile
    vol = kwave.get_Vol()
    ax = etc.newfig("Volume above fc [cm]", "time", "volume [cm]")
    #ax = plt.gca()
    ax.plot(rch_gwt.index, 10 * vol['V'], label="Volume")
    ax.grid(True)

    print("Done")
    
    plt.show()
    
    # %%
