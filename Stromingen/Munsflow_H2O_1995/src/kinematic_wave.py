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


from src import NL_soils as snl

# %%
secpd = 86400  # Seconds per day
K_s = 0.02     # Saturated hydraulic conductivity (m/d)
theta_s = 0.4  # Saturated soil moisture content (m^3/m^3)
theta_r = 0.1  # Residual soil moisture content (m^3/m^3)
lambda_bc = 3.7 # Pore size distribution index (dimensionless)
epsilon = 3 + 2 / lambda_bc  # epsilon
z = np.linspace(0, 5, 51)  # Depth from 0 to 5 Meters
t = np.linspace(0, 100., 51)

class KWprofile():
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
                        ('z', '<f8'),      # Depth (m)
                        ('v', '<f8'),      # Velocity (m/d)
                        ('theta1', '<f8'), # Upstream soil moisture content (m^3/m^3)
                        ('theta2', '<f8'), # Downstream soil moisture content (m^3/m^3)
                        ])

    # Properties of the soil
    props = {'K_s': K_s, 'theta_s': theta_s, 'theta_r': theta_r, 'epsilon': epsilon}

    def __init__(self, soil: snl.Soil, z_theta: np.ndarray =None):
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
        
        if z < 500.:
            raise ValueError("z does not seem to be in cm.")
        
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
        prof = np.zeros(len(z_theta), dtype=KWprofile.profile_dtype)
        
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
                      
                # Update the profile at the shock time
                # Note that every record of the profile caries its time, which equals global time    
                # Thus each profile record stands on its own, carries all required point data.
                self.profile['z'] += (tsh[ip_min] - self.profile['t']) * self.profile['v']
                self.profile['t'] = tsh[ip_min] # Update record times to current shock time
                
                # Set upstream theta of overtaking point to downstream theta of overtaken point               
                self.profile['theta2'][ip_min] = self.profile['theta2'][ip_min + 1]
                
                # Update front velocity of shock point
                self.profile['v'][ip_min] = self.point_velocities(
                    self.profile['theta1'][ip_min], self.profile['theta2'][ip_min])
                
                # Remove the point that was overtaken
                self.profile = np.delete(self.profile, ip_min + 1)                
            else:
                # No shocks in remaining this time step, just update the profile to t1
                self.profile['z'] += self.profile['v'] * (t1 - self.profile['t'])
                self.profile['t'] = t1
                break
        # self.profile = self.interpolate_between_profiles()
        return self.profile.copy()
    
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


    def prepend(self, q, Npt=15):
        sl = self.soil
        
        theta1 = sl.theta_fr_K(q)            
        theta2 = self.profile[0]['theta2']
        
        if theta2 > theta1:
            theta1 = np.linspace(theta1, theta2, Npt)
        theta2 = theta1
        
        head = np.zeros(len(np.atleast_1d[theta1]), dtype=KWprofile.profile_dtype)
        head['t'] = t
        head['v'] = self.point_velocities(theta)
        head['theta1'] = theta1
        head['theta2'] = theta2
        head['z']      = self.z0
        return np.prepend(head, self.profile)
        
    
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
        recharge['qwt'] = 0.
        time = (recharge.index - recharge.index[0]) / np.timedelta64(1, 'D')
        dt = np.diff(time)[0]
        
        recharge = pd.to_records(recharge)        
        
        for i, (t, pe) in enumerate(zip(time, recharge)):
            # Next value for flux from root zone
            q0 = pe['RCH']
            
            # Prepend to profile
            self.profile = self.prepend(q0)
            
            # Update the profile
            self.profile = self.update(dt)
            
            # Store profile for later animation
            self.profiles[t] = self.profile.copy()
            
            # Get flux at water table
            qwt, i2 = self.q_at_z(z)
            recharge['qwt'] = qwt
            
            # Truncate profile if more than one point beyond it            
            if len(self.profile) > i2:
                self.profile = np.delete(self.profile, slice(i2 + 1, None, 1))
                

        return pd.DataFrame(recharge)
      

    def interpolate_between_profiles(self, dz_min=0.1):
        """Interpolate between profiles to get a continuous smooth wave between the profiles.
        """
        z = self.profile['z']
        theta1 = self.profile['theta1']
        theta2 = self.profile['theta2']        
        t = self.profile['t']
        
        # first find the profiles, which are indicated by theta1 != theta2

        profile_indices = np.unique(
            np.hstack((0, np.where(theta1 != theta2)[0], len(theta1) - 1))
        )
        # Loop through the profile indices and interpolate between them
        new_profile = self.profile[:profile_indices[0] + 1].copy()  # Start with the first profile point
        for fi1, fi2 in zip(profile_indices[:-1], profile_indices[1:]):
            if np.max(np.diff(z[fi1:fi2 + 1])) < 2 * dz_min:
                new_profile = np.append(new_profile, self.profile[fi1 + 1:fi2 + 1])
                continue
            else:
                z2 = np.linspace(z[fi1], z[fi2], int((z[fi2] - z[fi1]) / dz_min) + 1)
                theta = theta1[fi1:fi2 + 1]
                theta[0] = theta2[fi1]
                theta_interp = np.interp(z2, z[fi1:fi2 +1], theta)

                # Interpolate the velocity
                f = np.zeros(len(z2) - 1, dtype=KWprofile.profile_dtype)
                f['z'] = z2[1:]
                f['theta1'] = theta_interp[1:]
                f['theta2'] = theta_interp[1:]
                f['t'] = t[fi1]
                f['v'] = self.point_velocities(theta_interp[1:], theta_interp[1:])
                f[-1] = self.profile[fi2].copy()
                new_profile = np.append(new_profile, f)
        return new_profile

    def fr_update_all(self, t):
        """Update the profile for all time steps."""
        self.profile['t'] = t[0]
        self.profiles = {t[0]: self.profile.copy()}
        Dt = np.diff(t)
        for t0, dt in zip(t[:-1], Dt):
            self.profiles[t0 + dt] = self.update(t0, dt)
        return self.profiles
    

    def plot_profile(self, profile=None, ax=None):
        """Plot the profile."""
        if profile is None:
            profile = self.profile
        t = profile['t'][0]
        z = np.vstack((profile['z'], profile['z'])).T.flatten()
        theta = np.vstack((profile['theta1'], profile['theta2'])).T.flatten()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.set_title('Kinematic Wave Profile')
            ax.set_xlabel('Depth (m)')
            ax.set_ylabel('Soil Moisture Content (m^3/m^3)')
            ax.grid(True)                
            ax.plot(z, theta, label=f't={t:.3f} d')
            ax.legend()
            plt.show()
        else:
            ax.plot(z, theta, label=f't={t:.3f} d')

    def plot_profiles(self, ax=None):
        """Plot the profiles."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        for t, profile in self.profiles.items():
            self.plot_profile(profile, ax=ax)
        ax.set_title('Kinematic Wave Profiles')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Soil Moisture Content (m^3/m^3)')
        ax.grid(True)
        ax.legend()

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    # Set up the plot style    
    plt.rcParams.update({
        'font.size': 12,
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'grid.alpha': 0.5,
        'lines.linewidth': 2,
        'lines.markersize': 5
    })

    # %% Setup the example usage
    
    # Simulate the flow through the unsaturated zone using the kinematic wave model.
    # The input is a time series of recharge on a daily basis.
    # When this works a root-zone module will be inserted to simulate the storage of water in the root zone.
    
    # Initialize the Kinematic Wave profile
    z_gwt = 5.  # Water table depth (m)   
    z = np.hstack((np.linspace(0,0.1, 51), np.linspace(0.2, z_gwt, 201)))
    
    
        
     
    t = rch['t'] # Time steps for the simulation
    
    # Animate the profile with time, showing kinematic waves in action
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_title('Kinematic Wave Profiles')
    ax.set_ylabel('Depth z (m)')
    ax.set_xlabel(r'Soil Moisture Content ($m^3 / m^3$)')
    ax.grid(True)
    ax.set_xlim(0., 1.25 * kw_profile.props['theta_s'])
    ax.set_ylim(z[-1], 0)
    
    # Remove the first point, it will be replaced by the input from recharge at each time step
    kw_profile.profile = np.delete(kw_profile.profile, [0])
    
    # Get the time, and z, theta of the current profile
    t, z, theta = kw_profile.tztheta


    
    # Set up the plot
    line, = ax.plot(theta, z, '.-', label=f"t={rch['t'][0]:.2f} d")
    txt = ax.text(0.6, 0.95, f't={0:.3f} d', transform=ax.transAxes)
    ax.grid(True)

    def update_profile(it):
        """Update the profile by prepending a new point and updating it to after the time step."""
        dt = rch['t'][it + 1] - rch['t'][it]
        kw_profile.prepend(rch['theta'][it], nNew=15)
        kw_profile.update(rch['t'][it], dt)  # Update the profile for the current time(s)
        rch['q_out'][it] = kw_profile.recharge(z_gwt=z_gwt)
        return

    def update(frame):
        """Animation function, updates artists at every new frame."""
        update_profile(frame)
        t, z, theta = kw_profile.tztheta   # Get the new profile
        line.set_ydata(z)
        line.set_xdata(theta)
        txt.set_text(f't={t:.3f} d')
        return line, txt

    ani = FuncAnimation(fig, update, frames=len(rch) - 1, blit=False)
    
    ani.save('animation.mp4', writer=FFMpegWriter(fps=10))
    
    # FF = F.fr_update_all(t)  # Update profiles for each time step
    # F.plot_profiles()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("In- and output of the unsaturated column")
    ax.set_xlabel('tijd [d]')
    ax.set_ylabel('flux [m/d]')
    ax.set_ylim(0., soil.k_from_theta(soil.theta_s))
    ax.grid(True)
    
    ax.plot(rch['t'], rch['q'], label="In: from root zone at z = {z[0]:.3f} m")
    ax.plot(rch['t'], rch['q_out'], label=f'Uit: recharge at gwt, z={z_gwt:.3f} m')
    
    ax.legend(loc='upper right')
    plt.show()
