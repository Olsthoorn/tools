# %% [markdown]
# Kinematic wave of moisture through unsaturated zone
#
# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
                        ('theta1', '<f8'), # Upstream soil moisture content (m^3/m^3)
                        ('theta2', '<f8'), # Downstream soil moisture content (m^3/m^3)
                        ('z', '<f8'),      # Depth (m)
                        ('t', '<f8'),      # Time (d)
                        ('v', '<f8')]      # Velocity (m/d)
                       )

    # Properties of the soil
    props = {'K_s': K_s, 'theta_s': theta_s, 'theta_r': theta_r, 'epsilon': epsilon}

    def __init__(self, theta=None, z=None, scen=3):
        """Initialize the kinematic wave profile using given soil properties and initial conditions, while choosing a scenario."""
        
        self.profile = np.zeros(len(z), dtype=KWprofile.profile_dtype)
        
        if theta is None:
            theta = self.theta_0(z, scen=scen)  # Default initial condition
        if len(theta) != len(z):
            raise ValueError("Length of theta must match length of z.")
        
        self.profile['theta1'] = theta
        self.profile['theta2'] = theta
        self.profile['z'] = z
        self.profile['t'] = 0.
        self.profile['v'] = self.vf(theta, theta)
        self.z0 = z[0]

    def theta_0(self, z, scen=1):
        """Initial condition for theta given a scenario."""
        if scen == 1: # Replace with actual condition
            theta_s = self.props['theta_s']
            theta_r = self.props['theta_r'] 
        
            return (1.1 * self.props['theta_r'] + 
                    0.9 * (self.props['theta_s'] -
                     self.props['theta_r']) * np.fmax(0,
                                np.sin(np.linspace(0, 4 * np.pi, len(z))))
            )
        elif scen == 2:  # Replace with actual condition
            theta = self.props['theta_r'] * np.ones(len(z))
            iz1 = int(0.25 * len(z))
            iz2 = int(0.75 * len(z))
            theta[iz1:iz2] = self.props['theta_s']
            return theta
        elif scen==3:  # Replace with actual condition
            theta = self.props['theta_r'] * np.ones(len(z))
            iz1 = int(0.10 * len(z))
            iz2 = int(0.20 * len(z))
            theta[iz1:iz2] = self.props['theta_s']
            theta[0:iz1] = np.linspace(self.props['theta_r'], self.props['theta_s'], iz1)
            return theta
        elif scen == 4:  # Replace with actual condition
            theta_r = self.props['theta_r']
            theta_s = self.props['theta_s']
            theta = theta_s * np.ones(len(z))
            iz1 = int(0.48 * len(z))
            iz2 = int(0.52 * len(z))            
            theta[:iz1] = np.linspace(theta_r, self.props['theta_s'], iz1)
            theta[iz2:] = np.linspace(self.props['theta_s'], theta_r, len(z) - iz2)
            return theta
        else:
            return np.linspace(theta_s, theta_r, len(z))

    def theta_from_q(self, q):
        """Calculate the soil moisture content from the flux q."""
        K_s = self.props['K_s']
        theta_r = self.props['theta_r']
        theta_s = self.props['theta_s']
        epsilon = self.props['epsilon']

        # Calculate the soil moisture content based on the flux
        theta = theta_r + (theta_s - theta_r) * (q / K_s) ** (1 / epsilon)
        return np.clip(theta, theta_r, theta_s) 
    
    def q_from_theta(self, theta):
        """Return downward flux q given the soil moisture content theta."""
        K_s = self.props['K_s']
        theta_r = self.props['theta_r']
        theta_s = self.props['theta_s']
        epsilon = self.props['epsilon']

        # Calculate the flux based on the soil moisture content
        q = K_s * ((theta - theta_r)  / (theta_s - theta_r)) ** epsilon
        return np.clip(q, 0, None)

    def vf(self, theta1, theta2):
        """Return the wave points velocities alsow when points are fronts.

        Parameters
        ----------
        theta1 : float
            Upstream soil moisture content 1 (m^3/m^3).
        theta2 : float
            Downstream soil moisture content 2 (m^3/m^3).
        epsilon : float, optional
            Pore size distribution parameter, by default 3.5.
        Returns
        -------
        float
            Velocity of the wave profile (m/d).
        """
        K_s = self.props['K_s']
        theta_s = self.props['theta_s']
        theta_r = self.props['theta_r']
        epsilon = self.props['epsilon']

        theta1, theta2 = np.atleast_1d(theta1, theta2)  # Ensure inputs are arrays
        theta1 = np.fmax(theta1, theta_r)
        theta2 = np.fmax(theta2, theta_r)
        theta1 = np.fmin(theta1, theta_s)
        theta2 = np.fmin(theta2, theta_s)
        
        v = np.zeros_like(theta1)
        L = np.isclose(theta1, theta2)

        v[L] = epsilon * K_s / (theta_s - theta_r) * ((theta1[L] - theta_r) / (theta_s - theta_r)) ** (epsilon - 1)

        K1 = K_s * ((theta1[~L] - theta_r) / (theta_s - theta_r)) ** epsilon
        K2 = K_s * ((theta2[~L] - theta_r) / (theta_s - theta_r)) ** epsilon

        v[~L] = (K1  - K2) / (theta1[~L] - theta2[~L])
        return v.item() if v.size == 1 else v

    def t_shock(self, ztol=1e-6):
        """Return shock times, i.e. times when points take over the position of the next point.

        Negative values imply that no shock can ever occur, the successive points diverge over time.
        
        Positive times indicate that shocks will occur at these times in the future if conditions remain the same.
        Parameters
        ----------
        ztol : float, optional, default 1e-6
            prevents division by zero if velocities are the same, which happens
            when generating several new points at once at the same depth every
            time then the recharge q is less than its previous value.
        """
        fr = self.profile
        dv = fr['v'][:-1] - fr['v'][1:]
        dv[np.isclose(dv, 0)] = 1e-10  # Avoid division by zero
        tsh = fr['t'][0] + np.fmax(fr['z'][1:] - fr['z'][:-1], ztol) / dv                    
        return tsh
    
    def fr_update(self, t0, dt):
        """Update the profile based on the time step."""
        t1 = t0 + dt

        while True:
            tsh = self.t_shock() # tsh is the time when the shock occurs
            
            # Find the indices where shock times are between t0 and t1
            Ip = np.where(np.logical_and(tsh > t0, tsh < t1))[0]

            if len(Ip) > 0:  # There are shocks in this time step           
                # Find the first shock time between t0 and t1
                ip_min = Ip[np.where(tsh[Ip] == tsh[Ip].min())][0]
                      
                # Update the profile at the shock time     
                self.profile['z'] += (tsh[ip_min] - self.profile['t']) * self.profile['v']
                self.profile['t'] = tsh[ip_min]
                self.profile['theta2'][ip_min] = self.profile['theta2'][ip_min + 1]
                self.profile['v'][ip_min] = self.vf(
                    self.profile['theta1'][ip_min], self.profile['theta2'][ip_min])
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
        Returns
        -------
        tuple (t, z, theta)
            t: the time of the profile (scalar)
            z: z of the current points of the profile, double to match theta1 and theta2 of each point
            theta: theta along the profile, for each point (same z) theta1 and theta2
        """
        z_profile = np.hstack((self.profile['z'], self.profile['z'])).T.flatten()
        theta_profile = np.hstack((self.profile['theta1'], self.profile['theta2'])).T.flatten()
        t_profile = np.mean(self.profile['t'])
        return t_profile, z_profile, theta_profile


    def interpolate_between_profiles(self, dz_min=0.1):
        """Interpolate between profiles to get a continuous smooth wave between the profiles."""
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
                f['v'] = self.vf(theta_interp[1:], theta_interp[1:])
                f[-1] = self.profile[fi2].copy()
                new_profile = np.append(new_profile, f)
        return new_profile

    def fr_update_all(self, t):
        """Update the profile for all time steps."""
        self.profile['t'] = t[0]
        self.profiles = {t[0]: self.profile.copy()}
        Dt = np.diff(t)
        for t0, dt in zip(t[:-1], Dt):
            self.profiles[t0 + dt] = self.fr_update(t0, dt)
        return self.profiles
    
    def prepend(self, theta, nNew=5):        
        """Prepend a new points to the existing profile."""
        if theta is None:
            return
        elif not np.isscalar(theta):
            raise ValueError("Input must be a scalar.")

        last = self.profile[0]
        if theta < last['theta1']:
            points = np.zeros(nNew, dtype=KWprofile.profile_dtype)
            points['t'] = last['t']
            points['z'] = self.z0
            points['theta1'] = np.linspace(theta, last['theta1'], nNew)
            points['theta2'] = np.linspace(theta, last['theta1'], nNew)
            points['v'] = self.vf(points['theta1'], points['theta2'])
        else:
            points = np.zeros(1, dtype=KWprofile.profile_dtype)
            points['t'] = last['t']
            points['z'] = self.z0
            points['theta1'] = theta
            points['theta2'] = theta
            points['v'] = self.vf(theta, theta)
        # Prepend the new points to the profile
        self.profile = np.append(points, self.profile.copy())
   
    def recharge(self, z_gwt):
        """Recharge leaving the profile at the water table, z_gwt."""        
        p = self.profile
        i1 = np.where(p['z'] < z_gwt)[0][-1]
        i2 = np.where(p['z'] >=  z_gwt)[ 0][0]
        theta = np.interp(z_gwt, (p['z'][i1], p['z'][i2]),
                          (p['theta2'][i1], p['theta1'][i2]))
        q = self.q_from_theta(theta)        
        # Truncate the profile beyond i2
        self.profile = np.delete(self.profile, slice(i2 + 1, None, 1))
        return q

    def plot_profile(self, fr, ax=None):
        """Plot the profile."""
        z = np.vstack((fr['z'], fr['z'])).T.flatten()
        theta = np.vstack((fr['theta1'], fr['theta2'])).T.flatten()
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
            ax.set_title('Kinematic Wave Profile')
            ax.set_xlabel('Depth (m)')
            ax.set_ylabel('Soil Moisture Content (m^3/m^3)')
            ax.grid(True)                
            ax.plot(z, theta, label=f't={fr['t'][0]:.3f}')
            ax.legend()
            plt.show()
        else:
            ax.plot(z, theta, label=f't={fr['t'][0]:.3f}')

    def plot_profiles(self, ax=None):
        """Plot the profiles."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        for t, fr in self.profiles.items():
            self.plot_profile(fr, ax=ax)
        ax.set_title('Kinematic Wave Profiles')
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Soil Moisture Content (m^3/m^3)')
        ax.grid(True)
        ax.legend()

if __name__ == "__main__":
    # This block is executed only when the script is run directly, not when imported.
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
    z_gwt = 05.  # Water table depth (m)   
    z = np.hstack((np.linspace(0,0.1, 51), np.linspace(0.2, z_gwt, 51)))
    kw_profile = KWprofile(theta=None, z=z, scen=4)  # Initialize profile with given scenario
    
    # Define the recharge time series
    q_dtype=np.dtype([('t', '<f8'), ('q', '<f8'), ('theta', '<f8'), ('q_out', '<f8')])
    rch = np.zeros(100, dtype=q_dtype)

    np.random.seed(42)  # For reproducibility
    rch['q'] = np.clip(np.random.uniform(-0.02, kw_profile.props['K_s'], size=100), 0, None)  # Random recharge values (m/d)
    rch['t'] = np.arange(100)  # Time in days
    rch['theta'] = kw_profile.theta_from_q(rch['q'])  # Calculate initial soil moisture
    rch['q_out'] = 0. # The flux leaving the profile at the water table
     
    t = rch['t'] # Time steps for the simulation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title('Kinematic Wave Profiles')
    ax.set_xlabel('Depth z (m)')
    ax.set_ylabel('Soil Moisture Content (m^3/m^3)')
    ax.grid(True)
    
    t, z, theta = kw_profile.tztheta
    line, = ax.plot(z, theta, label=f"t={rch['t'][0]:.2f} d")

    def update_profile(i):
        dt = rch['t'][i + 1] - rch['t'][i]
        kw_profile.prepend(rch['theta'][i], nNew=5)
        kw_profile.fr_update(rch['t'][i], dt)  # Update the profile for the current time(s)
        rch['q_out'][i] = kw_profile.recharge(z_gwt=z_gwt)
        return


    def update(frame):
        update_profile(frame)
        t, z, theta = kw_profile.tztheta        
        line.set_xdata(z)
        line.set_ydata(theta)
        ax.legend(loc='lower left')
        ax.relim()
        ax.autoscale_view()
        return line

    ani = FuncAnimation(fig, update, frames=len(rch) - 1, blit=False)
    
    ani.save('animation.mp4', writer=FFMpegWriter(fps=10))
    plt.close()
    
    # FF = F.fr_update_all(t)  # Update profiles for each time step
    # F.plot_profiles()
    plt.show()
