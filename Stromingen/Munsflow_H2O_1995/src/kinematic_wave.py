# %% [markdown]
# Kinematic wave of moisture through unsaturated zone
#
# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation, FFMpegWriter
import etc

cwd = os.getcwd()
dirs = etc.Dirs(cwd)
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from src.NL_soils import Soil #noqa

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
    
    profile_dtype = np.dtype([
                        ('t', '<f8'),      # Time (d)
                        ('tst1', '<f8'),   # start time of point (upstream side)
                        ('tst2', '<f8'),   # start time of point (downstream side)
                        ('z', '<f8'),      # Depth (m)
                        ('v', '<f8'),      # Velocity (m/d)
                        ('theta1', '<f8'), # Upstream soil moisture content (m^3/m^3)
                        ('theta2', '<f8'), # Downstream soil moisture content (m^3/m^3)
                        ])

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
        self.profiles={} # Store intermediate profiles for later animation
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
        
        if not isinstance(z_theta, np.ndarray) \
            or z_theta.dtype != self.__class__.z_theta_dtype:
            raise TypeError(f"z_theta must be an np.ndarray of dtype: {self.__class__.z_theta_dtype}")
        
        self.z0 = z_theta['z'][0]
        
        # start with t, tst1 and tst2 all zero        
        prof['t'] = 0.
        prof['z'] = z_theta['z']
        prof['theta1'] = self.soil.theta_fc()
        prof['theta2'] = self.soil.theta_fc()        
        prof['v'] = self.soil.dK_dtheta(self.soil.theta_fc())
        
        # Set tst1 and tst2 such that theta will be compute equal to theta_fc
        prof['tst1'] = -(prof['z'] - self.z0) / prof['v']
        prof['tst2'] = -(prof['z'] - self.z0) / prof['v']
        
        return prof

    def theta_from_q(self, q):
        """Calculate the soil moisture content from the flux q."""
        return self.soil.theta_fr_S(self.soil.S_fr_K(q))
    
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
        L = np.isclose(th1, th2)

        # Velocoty of normal points (no sharp fronts)
        if np.any(L):
            v[L] = self.soil.dK_dtheta(th1[L])

        # Veolocity of sharp fronts
        if np.any(~L):
            v[~L] = (self.soil.K_fr_theta(th1[~L]) - self.soil.K_fr_theta(th2[~L])
                 ) / (th1[~L] - th2[~L])
                
        return v.item() if v.size == 1 else v


    def get_shock_times(self, ztol=1e-6):
        """Return shock times, i.e. times when points overtake the next point.

        Negative values imply that no shock can ever occur, the successive points diverge over time.
        Positive times indicate that shocks will occur at these times in the future
        if conditions remain the same.
        
        Parameters
        ----------
        ztol : float, optional, default 1e-6
            prevents division by zero if velocities are the same, which happens
            when generating several new points at once at the same depth every
            time when the recharge q is less than its previous value.
        """
        prof = self.profile
        dv = prof['v'][:-1] - prof['v'][1:]
        dv[np.isclose(dv, 0)] = 1e-10  # Avoid division by zero
        
        tshock = prof['t'][0] + np.fmax(prof['z'][1:] - prof['z'][:-1], ztol) / dv                    
        return tshock
    

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
        t0 = self.profile['t'][0]
        t1 = t0 + dt # to start time, t1 end time, dt apart

        while True:
            tsh = self.get_shock_times() # tsh is the time when the shock occurs
            
            # --- Find the indices where shock times are between t0 and t1 ---
            Ip = np.where(np.logical_and(tsh > t0, tsh < t1))[0]

            if len(Ip) > 0:  # There are shocks in this time step           

                # --- Find the first shock time between t0 and t1
                ip_min = Ip[np.where(tsh[Ip] == tsh[Ip].min())][0]
                tshock = tsh[ip_min]
                      
                # --- Move points ahead to current shock time
                self.profile['z'] += (tshock - self.profile['t']) * self.profile['v']
                
                # --- Update profile records time to this smallest shock time
                self.profile['t']  =  tshock
                
                # --- Set upstream theta of overtaking point
                # --- to downstream theta of overtaken point               
                self.profile['theta2'][ip_min] = self.profile['theta1'][ip_min + 1]
                
                # --- Update front velocity of shock point
                self.profile['v'][ip_min] = self.point_velocities(
                    self.profile['theta1'][ip_min], self.profile['theta2'][ip_min])
                
                # --- Update start time of downstream side
                self.profile['tst2'][ip_min] = self.profile['tst2'][ip_min + 1]
                
                # --- Remove the point that was overtaken
                self.profile = np.delete(self.profile, ip_min + 1)
            else:
                # --- No shocks in remaining part of this time step
                # --- just update the profile to t1
                self.profile['z'] += self.profile['v'] * (t1 - self.profile['t'])
                self.profile['t'] = t1
                
                # --- leave the while loop
                break
        
        # --- Update the moisture content of all points.
        # --- this is relevant for fronts that
        # --- overtake tails and tails that intersect with downstream points.
        
        # --- The moisture content of each point is constant except at sharp fronts.
        
        prf = self.profile
        v_avg_1 = (prf['z'] - self.z0) / (prf['t'] - prf['tst1'])
        v_avg_2 = (prf['z'] - self.z0) / (prf['t'] - prf['tst2'])
        
        prf['theta1'] = self.soil.theta_fr_V(v_avg_1)
        prf['theta2'] = self.soil.theta_fr_V(v_avg_2)
        
        return self.profile
    

    def prepend(self, t, q, tol = 0.005):
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
        """
        
        previous = self.profile[0]
        theta1 = max(soil.theta_fc(), soil.theta_fr_K(q))        
                       
        if self.q0_prev > q + tol:     # tol in cm tol =0.005 cm = 0.05 mm
            # --- tail ----
            h = np.zeros(2, self.__class__.profile_dtype)
            
            theta2 = previous['theta1']
            
            h['theta1'] = np.array([theta1, theta2])
            h['theta2'] = np.array([theta1, theta2])
            h['tst1'] = t
            h['tst2'] = t
        elif self.q0_prev < q - tol:
            # --- shock ---
            h = np.zeros(1, self.__class__.profile_dtype)
            h['theta1'] = theta1
            h['theta2'] = previous['theta1']
            h['tst1']   = t
            h['tst2']   = previous['tst1']
        else:
            # --- normal points when q0_prev == q ---
            h = np.zeros(1, self.__class__.profile_dtype)
            h['theta1'] = theta1
            h['theta2'] = theta1
            h['tst1'] = t
            h['tst2'] = t
            
        h['z']  = self.z0
        h['t' ] = t
        h['v'] = self.point_velocities(h['theta1'], h['theta2'])
        
        # --- prepend h
        self.profile =np.concatenate([h, self.profile])

        return self.profile
        
    
    def q_at_z(self, z_gwt, dz=10):
        """Return the downward flux at z at current time.
        
        Can be used to get the recharge (flux through the water table at z_gwt)
        where z_gwt may vary over time
        
        """        
        p = self.profile

        # Points between which the z_gwt intersects
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

        # Flux at z_Gwt = K(theta_gwt)
        q_gwt = self.soil.K_fr_theta(theta_gwt)
        return q_gwt

    
    def get_profiles_obj(self, t: float, date: np.datetime64)-> dict:
        """Return dict that is added to the profiles for later plotting."""
        return {'profile': self.profile.copy(),
                    't':t,
                    'date': date,
                    'line': self.get_profile_line()}

        
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
        
        # --- Get time step size
        dt = np.diff(time)[0]

        # --- Initial storage of first profile (for later animation)
        self.profiles[0]=self.get_profiles_obj(time[0], date=recharge.index[0])
        
        # --- Convert pd.DataFrame to array with dtype of fields
        recharge = recharge.to_records()

        # --- remember first rootzone outflow
        self.q0_prev = recharge['RCH'][0]  / mm_per_cm # from mm/d --> cm/d    
        
        # --- Simulate timestep by timestep, record by record
        for ip, (t, pe) in enumerate(zip(time, recharge)):
        
            # --- Update the profile
            # --- evaluate shocktimes
            # --- move points over dt
            self.profile = self.update(dt)
            
            # --- next flux from root zone
            # --- (must be in cm/d)
            q0 = pe['RCH'] / mm_per_cm
            
            # --- Generate and prepend new record(s) to profile
            # if not np.isclose(q0, self.q0_prev, atol=0.005) or ip==0:
            self.profile = self.prepend(t + dt, q0)
            
            # --- Store curent profile for later animation ---
            self.profiles[ip + 1]=self.get_profiles_obj(
                            t + dt, date=pe['index'] + np.timedelta64(1, 'D'))
            
            # --- Get flux at water table
            qwt = self.q_at_z(self.z_gwt)
            pe['qwt'] = qwt * mm_per_cm # cm/d --> mm/d (store as mm/d)

            # --- clip points below z_gwt
            z = self.profile['z']
            Ip = np.arange(len(z))[z > z_gwt]
            if len(Ip) > 1:
                self.profile = self.profile[:Ip[1]]
            
            if self.q0_prev != q0:
                pass
            # --- remember previous q0
            self.q0_prev = q0
                            
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

            if np.isclose(p1['theta2'], p2['theta1']) or N < 3:
                # --- no interpolation needed
                z     += [z1, z2]
                theta += [p1['theta2'], p2['theta1']]
                # theta1, theta2 = self.soil.theta_fr_V((z1 - self.z0) / (t - p1['tst2'])), self.soil.theta_fr_V((z2 - self.z0) / (t - p2['tst1']))
            else:
                # --- interpolate N points including ends                
                z_ = np.linspace(z1, z2, N)
                v_avg  = (z_ - self.z0) / (t - p1['tst2'])
                theta_ = self.soil.theta_fr_V(v_avg)
                z     += list(z_)
                theta += list(theta_)
        return np.array(z), np.array(theta)
    

    def plot_profiles(self, ax=None):
        """Plot the profiles."""
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
    txt = ax.text(0.8, 0.3, f"t = {0.:.0f} d",
                   transform=ax.transAxes,
                   ha='center', va='center',
                   bbox=dict(
                        facecolor="gold",   # background color
                        edgecolor="black",  # border color
                        boxstyle="square"
                   )
                   )
    ax.legend()
    
    print("Progress:")

    # init_func: set up the initial state
    def init_func():        
        line.set_data([], [])
        txt.set_text('')        
        return (line, txt)
    
    # update_func: uses the closure to access `line` and `profiles`
    def update_func(frame):
        p = profiles[frame]
        (z, theta), date = p['line'], p['date']       
        line.set_xdata(z)
        line.set_ydata(theta)
        txt.set_text(f"{np.datetime64(date).astype('datetime64[D]')}")
        # --- show progress
        if np.mod(frame, 100) == 0:
            print('.', end="")
        if np.mod(frame, 1000) == 0:
            print(frame)
        return (line, txt)
    
    return fig, init_func, update_func

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


def change_meteo_for_testing(meteo: pd.DataFrame)-> pd.DataFrame:
    """Return meteo with adapted 'RCH' for easier testing and checking."""
    
    qrz = meteo['RCH'].values
    qrz = np.zeros(len(meteo), dtype=float)
    
    # --- Make five 10-day blocks of continuous q alternating between 0 and 1 mm/d ---
    N = 180
    for i in np.arange(0, len(meteo), 2 * N):
        qrz[i    :i + 1 * N] = 0.
        qrz[i + N:i + 2 * N] = 2. # mm/d
        
    meteo.loc[:, 'RCH'] = qrz
    return meteo
    
# %%

if __name__ == "__main__":
    
    # %% Setup the example usage
    
    # Simulate the flow through the unsaturated zone using the kinematic wave model.
    # The input is a time series of recharge on a daily basis.
    # When this works a root-zone module will be inserted to simulate the storage of water in the root zone.
    
    # First step get the recharge (leakage from root zone) from the meteo data.
    from src.rootz_rch_model  import RchEarth

    meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
    os.path.isfile(meteo_csv)
    deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)
    deBilt_short = deBilt.loc[deBilt.index >= np.datetime64("2020-01-01"), :]
        
    # Storage capacity
    Smax_I, Smax_R = 1.5, 100 # mm
    rch_simulator = RchEarth(Smax_I=Smax_I, Smax_R=Smax_R, lam=None)
    
    date_span = (np.datetime64("2020-01-01"), np.datetime64("2021-03-31"))
        
    rch = rch_simulator.simulate(deBilt_short)
    
    # --- replace rch DataFrame field ['RCH'] with a testing pattern
    if True:
        rch = change_meteo_for_testing(rch)

    # %% Second step get the soil and simulate the kinematic wave

    Soil.load_soils(os.path.join(dirs.data, "NL_VG_soilprops.xlsx"))
    soil = Soil('O05')
    
    # %%
    # Initialize the Kinematic Wave profile
    z_rz, z_gwt = 80, 2000.  # Water table depth (m)
    
    z_theta = np.zeros(10, dtype=Kinematic_wave.z_theta_dtype)
    z_theta['z'] = np.linspace(z_rz, z_gwt, 10)
    z_theta['theta'] = soil.theta_fc()

    kwave = Kinematic_wave(soil=soil, z_theta=z_theta) # z_GWT ?
    kwave.z_gwt = z_gwt
    
    rch_gwt = kwave.simulate(rch)
    
    fig, init_func, update_func = make_animation(kwave.profiles, soil, kwave.z_gwt)
    
    print("Running animation ...")
    ani = FuncAnimation(fig, update_func, frames=len(kwave.profiles), init_func=init_func,
                        blit=True, repeat=False)
    
    ani.save(f"Kinematic_wave_soil {soil.code}.mp4", writer="ffmpeg", fps=20)

    print("Done animation.")
    
    ax = etc.newfig("qwt", "time", "q cm/d")
    ax.plot(rch_gwt.index, rch_gwt['RCH'], label='qrtz')
    ax.plot(rch_gwt.index, rch_gwt['qwt'], label='qwt')
    ax.legend()
    ax.figure.savefig(f"qwt_{soil.code}")

    print("Done")
    
    # plt.show()
    
    # %%
