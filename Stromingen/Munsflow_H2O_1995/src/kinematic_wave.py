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
import functools
import etc

cwd = os.getcwd()
dirs = etc.Dirs(cwd)
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from src import NL_soils #noqa

plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 5
})

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

    z_theta_dtype = np.dtype(('z', '<f8'), ('theta', '<f8'))
    
    def __init__(self, soil: NL_soils.Soil, z_theta: np.ndarray =None)-> None:
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
        self.profile = self.set_profile(z_theta)
        self.z0 = self.profile['z'][0]
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
            raise TypeError(f"z_theta must be an np.ndarray of dtype: {}")
        
        # start with t, tst1 and tst2 all zero        
        prof['t'] = 0.
        prof['tst1'] = 0.
        prof['tst2'] = 0.
        prof['z'] = z_theta['z']
        prof['theta1'] = z_theta['theta']
        prof['theta2'] = z_theta['theta']
        
        prof['v'] = self.soil.dK_dtheta(self.soil.theta_fc())        
        
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
        sl = self.soil
        theta_s = sl.theta_s
        theta_r = sl.theta_r
        
        th1, th2 = np.atleast_1d(th1, th2)  # Ensure inputs are arrays
        
        th1 = np.fmax(th1, theta_r)
        th2 = np.fmax(th2, theta_r)
        
        th1 = np.fmin(th1, theta_s)
        th2 = np.fmin(th2, theta_s)
        
        v = np.zeros_like(th1)
        L = np.isclose(th1, th2)

        # Velocoty of normal points (no sharp fronts)
        v[L] = self.sl.dK_dtheta(th1[L])

        # Veolocity of sharp fronts
        v[~L] = (sl.K_fr_theta(th1[~L]) - sl.K_fr_theta(th2[~L])) / (th1[~L] - th2[~L])
                
        return v.item() if v.size == 1 else v


    def shock_times(self, ztol=1e-6):
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
            tsh = self.shock_times() # tsh is the time when the shock occurs
            
            # Find the indices where shock times are between t0 and t1
            Ip = np.where(np.logical_and(tsh > t0, tsh < t1))[0]

            if len(Ip) > 0:  # There are shocks in this time step           

                # Find the first shock time between t0 and t1
                ip_min = Ip[np.where(tsh[Ip] == tsh[Ip].min())][0]
                      
                # Move points ahead to current shock time
                self.profile['z'] += (tsh[ip_min] - self.profile['t']) * self.profile['v']
                
                # Update profile records time to this smallest shock time
                self.profile['t']  =  tsh[ip_min]
                
                # Set upstream theta of overtaking point
                # to downstream theta of overtaken point               
                self.profile['theta2'][ip_min] = self.profile['theta2'][ip_min + 1]
                
                # Update front velocity of shock point
                self.profile['v'][ip_min] = self.point_velocities(
                    self.profile['theta1'][ip_min], self.profile['theta2'][ip_min])
                
                # Update start time of downstream side
                self.prfofile['tst2'][ip_min] = self.profile['tst2'][ip_min + 1]
                
                # Remove the point that was overtaken
                self.profile = np.delete(self.profile, ip_min + 1)                
            else:
                # No shocks in remaining this time step, just update the profile to t1
                self.profile['z'] += self.profile['v'] * (t1 - self.profile['t'])
                self.profile['t'] = t1
                break
        
        # Update the moisture concent of all points. It's relevant for fronts that
        # Overtake tails and tails that intersect with downstream points.
        
        # The moisture content of each point is constant except at sharp fronts.
        
        prf = self.profile
        v_avg_1 = (prf['z'] - self.z0) / (prf['t'] - prf['tst1'])
        v_avg_2 = (prf['z'] - self.z0) / (prf['t'] - prf['tst2'])
        
        prf['theta1'] = self.soil.theta_fr_V(v_avg_1)
        prf['theta2'] = self.soil.theta_fr_V(v_avg_2)
        
        return self.profile
    
    @ property
    def tztheta(self):
        """Return t, z, and theta of the current profile.
        
        A profile exists at a certain time and changes with time.
        
        Returns
        -------
        tuple (t, z, theta)
            t: float
                Time of the profile
            z: (float, float)
                z of the current points of the profile,
                double to match theta1 and theta2 of each point
            theta: (float, float)
                theta along the profile, for each point (same z) theta1 and theta2
        """
        z_profile = np.vstack((self.profile['z'],
                               self.profile['z'])).T.flatten()
        
        theta_profile = np.vstack((self.profile['theta1'],
                                   self.profile['theta2'])).T.flatten()
        
        t_profile = np.mean(self.profile['t'])
        
        return t_profile, z_profile, theta_profile


    def prepend(self, t, q, Npt=15):
        sl = self.soil
        
        theta1 = sl.theta_fr_K(q)            
        theta2 = self.profile[0]['theta1']
        
        # if theta2 > theta1:
        #     theta1 = np.linspace(theta1, theta2, Npt)
        # theta2 = theta1
        
        h = np.zeros(0, self.__class__.profile.dtype)
        h['t' ] = t
        h['tst1'] = t
        h['tst2'] = t        
        h['theta1'] = sl.theta_fr_K(q)
        h['theta2'] = self.profile[0]['theta1']
        h['v'] = self.point_velocities(h['theta1'], h['theta2'])
        h['z']      = self.z0
        
        self.profile = np.prepend(h, self.profile)

        return self.profile
        
    
    def q_at_z(self, z_gwt):
        """Return the downward flux at z at current time.
        
        Can be used to get the recharge (flux through the water table at z_gwt)
        where z_gwt may vary over time
        
        """        
        p = self.profile

        # Points between which the z_gwt intersects
        i1 = np.where(p['z'] < z_gwt)[0][-1]
        i2 = np.where(p['z'] >=  z_gwt)[ 0][0]

        # The moisture content at z_gwt
        theta = np.interp(z_gwt, (p['z'][i1], p['z'][i2]),
                          (p['theta2'][i1], p['theta1'][i2]))

        # The flux at z_Gwt
        q = self.soil.K_fr_theta(theta)        

        # Truncate the profile beyond i2 + 1
        if len(self.profile) > i2:
            self.profile = np.delete(self.profile, slice(i2 + 1, None, 1))
        return q

        
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
        
        # initialize or set column with recharge at the groundwater table
        recharge['qwt'] = 0.        
        
        # get time in days as floats from datetime index        
        time = (recharge.index - recharge.index[0]) / np.timedelta64(1, 'D')
        
        # Get time step size
        dt = np.diff(time)[0]
        
        # Convert pd.DataFrame to array with dtype of fields
        recharge = pd.to_records(recharge)        
        
        # Simulate timestep by timestep, record by record
        for t, pe in zip(time, recharge):

            # Next value for flux from root zone
            q0 = pe['RCH']
            
            # Gnerate and prepend new record(s) to profile
            self.profile = self.prepend(t, q0)
            
            # Update the profile, evluage shocktimes and move points over dt
            self.profile = self.update(dt)
            
            # Store profile for later animation
            self.profiles[t]['profile'] = self.profile.copy()
            self.profiles[t]['line'] = self.get_profile_line()
            
            # Get flux at water table
            qwt, i2 = self.q_at_z(z)
            pe['qwt'] = qwt
            
            # Truncate profile if more than one point beyond it 
            if False: 
                if len(self.profile) > i2:
                    self.profile = np.delete(self.profile, slice(i2 + 1, None, 1))
                
        return pd.DataFrame(recharge)
      

    def get_profile_line(self, dz_min=0.1):
        """Interpolate between profiles to get a continuous smooth wave between the sub profiles.
        
        We have a single large profile over the total depth of the percolation zone.
        Subprofiles are parts of the overall profile between sharp fronts. These
        subprofiles may be long but with only few points to actually define them.
        Therefore, to keep smooth subprofiles, we use this function to itnerpolate
        points between them. Perhaps the profile could be inspected for parts to
        gain more points.
                
        """
        prf = self.profile
        
        z, theta = [], []        
        for (p1, p2) in zip(prf[:-1], prf[1:]):
            N = 2 + np.int(p2['z'] - p1['z']) / 0.1
            if np.isclose(p1['theta2'], p2['theta1']) or N < 1:
                z += [p1['z'], p2['z']]
                theta += [p1['theta2'], p2['theta1']]                
            else:
                z_ = np.linspace(p1['z'], p2['z'], N)
                v_avg = z_ / (t - p1['tst_2'])
                theta_ = self.soil_theta_fr_V(v_avg)
                z.append(list(z_))
                theta.append(list(theta_))
        return np.array(z), np.array(theta)
    

    def plot_profiles(self, ax=None):
        """Plot the profiles."""
        if ax is None:
            ax = etc.newfig("Profile", "z [cm]", "theta", figsize=(10, 6))

        for t in self.fprofiles:
            z, theta = self.profiles[t]['line']
            ax.plot(z, theta, label=f"t = {:.3g} d")


def make_animation(profiles):
    fig, ax = plt.subplots()
    
    ax.set_title ='Kinematic Wave Profiles'
    ax.set(xlabel='z [cm]', ylabel='theta' )
    
    # Create the artists inside the factory
    line, = ax.plot([], [], lw=2)
    
    # init_func: set up the initial state
    def init_func():
        line.set_data([], [])
        return (line,)
    
    # update_func: uses the closure to access `line` and `profiles`
    def update_func(frame):
        z, theta = profiles[frame]['ztheta']
        line.set_xdata(z)
        line.set_ydata(theta)
        return (line,)
    
    return fig, init_func, update_func

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
 
    # %% Second step get the soil and simulate the kinematic wave

    soil = NL_soils('O01')
    
    # Initialize the Kinematic Wave profile
    z_rz, z_gwt = 80, 2000.  # Water table depth (m)
    
    z_theta = np.zeros((10, 2))
    z_theta[:, 0] = np.linspace(z_rz, z_gwt, 10)
    z_theta[:, 1] = soil.theta_fc()

    kwave = Kinematic_wave(soil=soil, z_theta=z_theta) # z_GWT ?
    
    kwave.simulate(rch)
    
    fig, init_func, update_func = make_animation(kwave.profiles)
    
    ani = FuncAnimation(fig, update_func, frames=50, init_func=init_func, blit=True)
    
    plt.show()
