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

import cProfile
import pstats

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
    The profile consists of points that represent the soil moisture content at different depths over time. Each point carries its time, depth, velocity and the upstream and downstream soil moisture content (thetaL and thetaR). When a point is a moisture front, the downstream soil moisture content is lower than the upstream soil moisture content.
    The velocity of the wave is calculated based on the soil moisture content and the saturated hydraulic conductivity.
    The soil moisture content is always between theta_r (residual) and theta_s (saturated).
    The initial condition for the soil moisture content can be set using different scenarios.
    The class provides methods to update the profile over time, calculate the velocity of the wave, and plot the profile or all profiles at once showing the development of the moisture front over time.

    @ TO 2025-08-01
    """
    
    # --- dtype of kinematic wave profile
    profile_dtype = np.dtype([
                        ('t', '<f8'),      # Time (d)
                        ('tstL', '<f8'),   # start time of point (upstream side)
                        ('tstR', '<f8'),   # start time of point (downstream side)
                        ('z', '<f8'),      # Depth (m)
                        ('v', '<f8'),      # Velocity (m/d)
                        ('vp', '<f8'),     # Velocity at end of previous time step
                        ('thetaL', '<f8'), # Upstream soil moisture content (m^3/m^3)
                        ('thetaR', '<f8'), # Downstream soil moisture content (m^3/m^3),
                        ('front', '?')     # indicats the point is a sharp front
                        ])

    # --- dtype of input data, i.e. of initial profile
    z_theta_dtype = np.dtype([('z', '<f8'), ('theta', '<f8')])
    
    
    def __init__(self, soil: Soil, z0: float =0, zwt: float =1000, N: int =10)-> None:
        """Initialize the kinematic wave profile using props in soil.
        
        Parameters
        ----------
        soil: soil.snl.Soil object
            A dutch soil with its methods and properties
        z0: float
            depth in cm of  base of root zone (= top of percolation zone)
        zwt: float
            depth in cm of base of groundwater table (= bottom of percolation zone)
            Because z is depth, zwt must be larger than z0
        N: int >= 2
            Number of points in initial moisture profile.
        """        
        self.soil = soil
        self.profile = self.initial_profile(z0=z0, zwt=zwt, N=N) # Also sets self.z0
        
         # --- Intermediate profiles for later animation
        self.profiles={}
        return None
        
    def initial_profile(self, z0: float =0, zwt: float =1000, N: int =10)->np.ndarray:
        """Return the initial moisture profile.
        
        A profile represents the moisture situation between the bottom
        of the root zone z0 and the water table zwt (zwt = the end of the percolation zone).

        The profile is an np.ndarray of dtype specified in the class (profile_dtype).
                
        Each record of the profile represents a (moisture) point, represented by
        the current time, its velocity, its theta and its launch time. The theta
        has a left upstream value thta1 and a right (downstream) value thetaR.
        It start time has a left upstream values tstL and a right downstream value tstR.
        Two values for theta and tst are needed to handle sharp fronts. Sharp fronts
        originate from points overtaking previously launche points and points being
        overtaken by later launched points.
        
        To handle decelleration of sharp fronts, we also remember the velocity of the
        previous time step.
        
        Finally a boolean 'front' is in the dtype for convenience to mark points that
        are or have become sharp fronts.
        
        The speed of a (moisture profile) point is uniquely determined by the combination
        of its upstream and downstream moisture content and the relation K(theta), or
        rather dK(theta)/dtheta pertaining to the soil.
        
        Returns
        -------
            Initial moisture profile, a np.array with the dtype
            specified in the class as profile_dtype.
            All values, except for z, will be set to default values here:
            * thetaL and thetaR --> self.soil.theta_fc()
            * tstL and tstR --> np.nan
            * v and vp --> dK/dthtea(theta_fc) (almost 0).
            * front --> False
        """
        N = np.fmax(N, 2)
        
        # --- Verify input of initial profile
        if zwt <= z0:
            raise ValueError("z water table must be larger than z root zone.")

        self.z0  = z0
        self.zwt = zwt
        
        prof = np.zeros(N, dtype=self.__class__.profile_dtype)
        
        # --- Start with t, tstL and tstR all zero 
        prof['z']      = np.linspace(z0, zwt, N)
        prof['t']      = 0.
        prof['thetaL'] = self.soil.theta_fc()
        prof['thetaR'] = self.soil.theta_fc()        
        prof['v']      = self.soil.dK_dtheta(self.soil.theta_fc())
        prof['vp']     = prof['v'].copy()  # v at end of previous time step
        prof['tstL']   = np.nan
        prof['tstR']   = np.nan
        
        return prof

    def point_velocities(self, thetaL: float | np.ndarray, thetaR: float | np.ndarray, tol=1e-4)-> float | np.ndarray:
        """Return the velocity of all the wave points velocities.

        Parameters
        ----------
        thL : float
            Upstream soil moisture content 1 (m^3/m^3).
        thR : float
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
        
        thetaL, thetaR = np.atleast_1d(thetaL, thetaR)  # Ensure inputs are arrays
        
        thetaL = np.fmin(theta_s, np.fmax(thetaL, theta_r))
        thetaR = np.fmin(theta_s, np.fmax(thetaR, theta_r))
        
        v = np.zeros_like(thetaL)

        # --- Velocity of points upstream and downstream of sharp front
        v_theta1 = np.atleast_1d(self.soil.dK_dtheta(thetaL))
        v_theta2 = np.atleast_1d(self.soil.dK_dtheta(thetaR))

        # --- Bool array separating sharp front points from ordinary points
        L = np.isclose(thetaL, thetaR, tol)
        
        # --- Velocity of normal points (no sharp fronts)
        if np.any(L):
            v[L] = v_theta1[L]

        # --- Veolocity of sharp fronts
        if np.any(~L):
            v[~L] = (self.soil.K_fr_theta(thetaL[~L]) - self.soil.K_fr_theta(thetaR[~L])
                 ) / (thetaL[~L] - thetaR[~L])
    
        return (v_theta1.item() if v_theta1.size == 1 else v_theta1,
                v.item()    if v.size    == 1 else v,
                v_theta2.item() if v_theta2.size ==1 else  v_theta2)


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
        
        # --- estimated veclocities, taking decelleration of front points into account
        # --- choose velocity at t + dt
        v = 1.5 * prof['v'] - 0.5 * prof['vp']
        
        # --- Avoid division by zero for points that happen to have the same z at given time
        dv = np.fmax(vtol, v[:-1] - v[1:])
        
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
        
        # --- Prevents repeating computing thetaL and thetaR in Runge Kutta step
        def update_theta(ip, z, prev=None, cur=None, next_=None):
            """Return update thetaL and thetaR computed from the front position
            and wheather left and or right of the front is a tail, which is
            tested using tst1 and tst2 of both the current point and its
            left and right neighbors
            """
            thL, thR = cur['thetaL'], cur['thetaR']
            if ip > 0:
                if prev['tstR'] <= cur['tstL']: # tail to the left
                    thL = self.soil.theta_fr_V((z - z0) / (t_next - tstL))
            if ip < len(self.profile) - 1:
                if cur['tstR'] <= next_['tstL']: # tail to the right
                    thR = self.soil.theta_fr_V((z - z0) / (t_next - tstR))
            return thL, thR

        # --- first estimate of the new point positions and velocities

        t = self.profile['t'][0]
        dt = t_next - t
        z0 = self.z0
                
        # --- for front points:
        #           adapt theta to new theta and v to mean v over new dt.
        #           but only of left is a tail and or right is a tail
        #           else theta and velocity does not change
        # --- non-front points keep their velocity
        # --- Last point in the profile gets velocity 0 (stays put at zwt)
        
        # --- When prev[tstR] > (younger) cur[tstL] no change, no change of cur[thetaL], no tail at left
        # --- When cur['tstR] > (younger) next_[tstR], no change of cur[thetaR], not a tail at the right
        
        # --- Front points
        L = self.profile['front']
        
        # ---Points to move separately        
        Ifr = np.where( L)[0] # --- Front points, move using RK if left and/or  right is a tail
        Iot = np.where(~L)[0] # --- Other points just move the current v dt
        
        # --- The front points to move using Runge Kutta preditor corrector (2nd order)
        # --- this is because thetaL and thetaR and, thefore v, change during time step
        for ip in Ifr:            

            # --- Situation at the current time for the front points
            cur = self.profile[ip]
            z1     = cur['z']
            v1     = cur['v'] 
            tstL   = cur['tstL']          
            tstR   = cur['tstR']
                            
            # --- left (1) and right (2) of front represent points launched at times tstL and tstR resp.
            # --- Estimate Left and Right velocities at t_next since Left and Right sides of front were launched

            # --- get left neighbor (younger point) if it exists            
            if ip > 0:
                prev = self.profile[ip - 1]
            else:
                prev = None
                
            # --- get right neighbor (older point) if it exists
            if ip < len(self.profile) - 1:
                next_ = self.profile[ip + 1]
            else:
                next_ = None

            # --- At end position recompute thetaL and thetaR            
            thL, thR = update_theta(ip, z1 + v1 * dt, prev, cur, next_)

            # --- and compute the accompanying front velocity
            v2 = self.point_velocities(thL, thR)[1]
            
            # --- then recompute the end point at t_next using average v
            ze = z1 + 0.5 * dt * (v1 + v2)                                        

            # --- At end position recompute thetaL and thetaR            
            thL, thR = update_theta(ip, ze, prev, cur, next_)

            # --- and the new front velocities at the end of the time step
            ve = self.point_velocities(thL, thR)[1]

            # --- remember previous velocity to allow computing decelleration of fronts when computing shock times
            self.profile['vp'][ip] = self.profile['v'][ip]
            
            # --- Fill in the new values at the end position
            self.profile['thetaL'][ip] = thL
            self.profile['thetaR'][ip] = thR
            self.profile['z'][ip] = ze
            self.profile['v'][ip] = ve
            self.profile['t'][ip] = t_next
            
        # --- Forward all other points using their current constant velocities
        if Iot.size > 0:

            # --- Handle last point in profile (zwt) which does not move
            # --- always
            self.profile['v'][-1] = 0.
         
            self.profile['z'][Iot] += self.profile['v'][Iot] * dt
            self.profile['t'][Iot]  = t_next
            
        # --- points may not overtake each other during a front step
        z = self.profile['z']
        if np.any(z[:-1] > z[1:]):                        
            Ip = np.where(z[:-1] > z[1:])[0]
            z[Ip], z[Ip + 1] = z[Ip + 1].copy(), z[Ip].copy()
            pass
            
                    
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
            tshock = tsh[ip_min]

            if tshock <= t1:
                # --- We forward all points to time = tshock
                self.profile['front'][ip_min]  = True
                                          
                # --- Also forward all points to time t_shock
                self.front_step(tshock)
                                
                # --- Shock now happening: handle it
                # --- Copy downstream properties of overtaken point to overtaking point
                self.profile['thetaR'][ip_min] = self.profile['thetaR'][ip_min + 1]
                self.profile['tstR'][ip_min]   = self.profile['tstR'][ip_min + 1]
                
                # --- Adapt the velocity of the shock point (overtaking point)
                self.profile['v'][ip_min]      = self.point_velocities(self.profile['thetaL'][ip_min], self.profile['thetaR'][ip_min])[1]  
                
                # --- Acknowledge that this is a shock point
                self.profile['front'][ip_min] = True
                             
                # --- Remove the point that was overtaken              
                self.profile = np.delete(self.profile, ip_min + 1)
                
            else:
                # --- No shocks in (remaining part of) this time step
                # --- just update the profile to t1
                self.front_step(t1)
                
                # --- leave the while loop
                break
                
        return None
    

    def prepend(self, t, q, N=15, tol = 0.0001):
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
        thetaL = max(soil.theta_fc(), soil.theta_fr_K(q))
        thetaR = previous['thetaL']    
        
        # --- a front is when theta of current point > theta previous point
        front  = thetaL > thetaR + tol
        same_q = np.isclose(thetaL, thetaR, tol)

        if same_q:
            # --- normal points when q0_prev == q ---
            current = np.zeros(1, self.__class__.profile_dtype)
            current['thetaL'] = thetaL
            current['thetaR'] = thetaL
            current['tstL'] = t
            current['tstR'] = t
        elif front:   # tol in cm tol =0.005 cm = 0.05 mm
            # --- shock, sharp front ----
            current = np.zeros(1, self.__class__.profile_dtype)
            current['thetaL'] = thetaL
            current['thetaR'] = thetaR
            current['tstL'] = t
            current['tstR'] = previous['tstL']
            current['front'] = True
        else:
            # --- tail, new points are generated at the same time and z ---
            N = max(1, int((thetaR - thetaL) * 1000))
            N = 2
            current = np.zeros(N, self.__class__.profile_dtype)
            thetas = np.linspace(thetaL, thetaR, N)
            current['thetaL'] = thetas
            current['thetaR'] = thetas
            current['tstL']   = t
            current['tstR']   = t
            
        current['z']  = self.z0
        current['t']  = t        
        current['v'] = self.point_velocities(current['thetaL'], current['thetaR'])[1]
        current['vp'] = current['v'].copy()
        
        # --- prepend h
        if ((len(self.profile) > 1) and
            np.isclose(self.profile['thetaL'][0], self.profile['thetaR'][0], tol) and
            np.all(np.isclose(self.profile['thetaL'][0], current['thetaR'],         tol)) and            
            np.isclose(self.profile['thetaR'][0], self.profile['thetaL'][1], tol)):            
            # --- Don't need previous point if q equals that of the current point.
            self.profile = np.concatenate([current, self.profile[1:]])
        else:
            # --- q different from previous point.
            self.profile = np.concatenate([current, self.profile])

        # --- Don't return anything
        return None

    
    def q_at_z(self, zwt, dz=10):
        """Return the downward flux at z at current time.
        
        Can be used to get the flux through the water table at zwt.
        Works even with zwt varying over time.
        
        """        
        p = self.profile

        # --- Points between which the zwt intersects
        i1 = np.where(p['z'] < zwt)[0][-1]
        i2 = np.where(p['z'] >=  zwt)[ 0][0]
        p1, p2 = p[i1], p[i2]

        if np.isclose(p1['thetaR'], p2['thetaL'], rtol=1e-3):
            theta_gwt = p1['thetaR']            
        else: # Interpolate the moisture content at zwt       
            z1, z2 = p1['z'], p2['z'] 
            N = int(2 + (z1  -z1) / dz)
            z = np.linspace(z1, z2, N)
            v_avg = (z - self.z0) / (p1['t'] - p1['tstR'])
            theta = self.soil.theta_fr_V(v_avg)
            theta_gwt = np.interp(zwt, z, theta)

        # --- Flux at z_Gwt = K(theta_gwt)
        q_gwt = self.soil.K_fr_theta(theta_gwt)
        return q_gwt

    
    def get_profiles_obj(self, t: float, date: np.datetime64)-> dict:
        """Return dict that is added to the profiles for later plotting.
        
        This function is called in the simulation loop and works
        directly on self.profiles.
        
        t and date are stored and not used.
        
        t will match self.profile['t']
        """
        if t == 74:
            pass

        # --- get the interpolated z, theta of the profile
        # --- this line will be used in animation
        z, theta = self.get_profile_line()
        
        # --- last z where z < zwt
        iz = np.where(z <= self.zwt)[0][-1]
        
        # --- interpolate to get theta at exactly z = zwt
        theta_iz = np.interp(self.zwt, z[iz:], theta[iz:])
        
        # --- Make the point (zwt, theta_iz) the last point of the profile
        z     = np.hstack((z[:iz], self.zwt))
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
        # --- The very first record has z=self.z0, t=0, tstL=0, tstR=0, than theta_fc, theta_fc
        for ip, (t, pe) in enumerate(zip(time, recharge)):
        
            if np.mod(ip, 74) == 0:
                pass

            # --- next flux from root zone
            # --- (must be in cm/d)
            q0 = pe['RCH'] / mm_per_cm
                
            # --- Generate and prepend new record(s) to profile            
            self.prepend(t, q0)

            # --- Store current profile for later animation ---
            # --- Store profiel at time = t (at the beginning of the day)
            self.profiles[ip]=self.get_profiles_obj(t, date=pe['index'])
            
            theta_gwt = self.profiles[ip]['line'][1][-1]
            pe['qwt'] = self.soil.K_fr_theta(theta_gwt) * mm_per_cm

            # --- Update the profile to end of the day (t1 = t + dt)
            # --- evaluate shocktimes
            # --- move points over dt
            self.update(dt)
                    
            # --- Get flux at water table
            # --- Covered in get_profiles_obj, where theta_gwt = last theta in profile line
            #qwt = self.q_at_z(self.zwt)
            
            # --- clip points below zwt
            # --- Also covered in get_profiles_obj
            # z = self.profile['z']
            # Ip = np.arange(len(z))[z > zwt]
            # if len(Ip) > 1:
            #     self.profile = self.profile[:Ip[1]]
        
        # --- Store profile at end of the last day
        # --- Get last time step size
        dt = np.diff(time)[-1]
        self.profiles[ip]=self.get_profiles_obj(t + dt, date=pe['index'] + np.timedelta64(1, 'D'))
                            
        return pd.DataFrame(recharge)
      

    def get_profile_line(self, dz=10, tol=0.001):
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

            # --- If the two thetas are the same, constant theta, --> no interpolation necessary
            if np.isclose(p1['thetaR'], p2['thetaL'], tol) or N < 3:
                z     += [z1, z2]
                theta += [p1['thetaR'], p2['thetaL']]
                continue
            else:
                # --- interpolate N points including the ends                
                z_ = np.linspace(z1, z2, N)
                v_avg  = (z_ - self.z0) / (t - p1['tstR'])
                theta_ = self.soil.theta_fr_V(v_avg)
                                
                # --- Check and compare with the profile
                # --- Use the profile values at the ends of the itnerpolated points
                # --- They should match the interpolated values
                theta_[[0, -1]] = p1['thetaR'], p2['thetaL']
                z     += list(z_)
                theta += list(theta_)
                
        # --- End the curve at the groundwater table
        z, theta = np.array(z), np.array(theta)
        I_ =[np.where(z <  self.zwt)[0][-1], np.where(z >= self.zwt)[0][ 0]]
        
        # --- Get theta at zwt
        theta_gwt = np.interp(self.zwt, z[I_], theta[I_])
        
        # --- Generate the curve ending on exactly zwt
        mask = z < self.zwt
        z     = np.hstack((z[    mask], self.zwt))
        theta = np.hstack((theta[mask], theta_gwt))  
                
        return np.array(z), np.array(theta)


    def plot_profiles(self, ax=None):
        """Plot the profiles on given axes."""
        if ax is None:
            ax = etc.newfig("Profile", "z [cm]", "theta", figsize=(10, 6))

        for ip in self.profiles:
            z, theta = self.profiles[ip]['line']
            t = self.profiles[ip]['t']
            ax.plot(z, theta, label=f"t = {t:.3g} d")

# %% --- Animation closure

def make_animation(profiles: dict, soil: Soil, zwt: float)->tuple:

    # --- determine extent of axes ---    
    z = profiles[0]['line'][0]
    zmin = z[0]
    zmax = z[-1]
    zmax = zwt
        
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
    points, = ax.plot([], [], 'ro')
    
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
        points.set_data([], [])
        txt.set_text('')        
        return (line, txt)
    
    # --- Update_func: uses the closure to access `line` and `profiles`
    def update_func(frame):
        p = profiles[frame]
        
        # --- get line and date
        (z, theta), date = p['line'], p['date']       
        line.set_xdata(z)
        line.set_ydata(theta)
        
        # --- add profile points for verification
        prof = p['profile']
        (z, thL, thR) = prof['z'], prof['thetaL'], prof['thetaR']
        points.set_xdata(np.vstack((z, z)).flatten())
        points.set_ydata(np.vstack((thL, thR)).flatten())
        
        txt.set_text(f"{np.datetime64(date).astype('datetime64[D]')}, t={frame} d")
        # --- show progress
        if np.mod(frame + 1, 100) == 0:
            print('.', end="")
        if np.mod(frame + 1, 1000) == 0:
            print(frame)
        return (line, txt)
    
    return fig, init_func, update_func

# %% --- extra functions

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
    
# %% =======================
if __name__ == "__main__":
    
    # %% --- Setup the example usage
    

    # Simulate the flow through the unsaturated zone using the kinematic wave model.
    # The input is a time series of recharge on a daily basis.
    # When this works a root-zone module will be inserted to simulate the storage of water in the root zone.
    
    with cProfile.Profile() as pr:
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
        if False:
            rch = change_meteo_for_testing(rch, N=50, m=1)[rch.index < rch.index[0] + np.timedelta64(300, 'D')]
            
            # --- Continuous infiltration from day 0
            # rch.loc[rch.index <= rch.index[0] + np.timedelta64(50, 'D'), 'RCH'] = 2 # mm/d

        # --- Second step get the soil and simulate the kinematic wave
        Soil.load_soils(os.path.join(dirs.data, "NL_VG_soilprops.xlsx"))

        # --- Choose a soil from the Staringreeks
        soil = Soil('O05')

        # --- Simulate and animate the kinematic wave over time
            
        # --- Initialize the Kinematic Wave profile

        # --- Root zone and gwroundwater table depth in cm
        z0, zwt = 0, 2000.

        # --- Initiate the kinematic wave object
        kwave = Kinematic_wave(soil=soil, z0=z0, zwt=zwt, N=2) # z_GWT ?
        kwave.profile['thetaL'] = kwave.soil.theta_fr_K(0.1)
        kwave.profile['thetaR'] = kwave.soil.theta_fr_K(0.1)

        # --- Simulate the kinematic wave
        rch_gwt = kwave.simulate(rch)

        # --- Setup of the animation    
        fig, init_func, update_func = make_animation(kwave.profiles, soil, kwave.zwt)

        # --- Animate
        if True:
            print(f"Running animation, showing progress, one dot per 100 frames, total number of frames: {len(rch_gwt)}")
            ani = FuncAnimation(fig, update_func, frames=len(kwave.profiles), init_func=init_func,
                                blit=True, repeat=False)

            # --- Save anaimation
            ani.save(f"Kinematic_wave_soil {soil.code}.mp4", writer="ffmpeg", fps=20)

            plt.close(fig)

            print("Done animation.")

        # --- plot RCH and qwt
        ax = etc.newfig(f"q at the water table, z0={kwave.z0} cm, zwt={kwave.zwt} cm", "time", "q mm/d")
        ax.plot(rch_gwt.index, rch_gwt['RCH'], label='qrtz')
        ax.plot(rch_gwt.index, rch_gwt['qwt'], label='qwt')
        ax.grid(True)
        ax.legend()
        ax.figure.savefig(f"qwt_{soil.code}")

        # --- plot integrated curve of RCH and qwt
        ax = etc.newfig(f"Flux q integrated over time from rootzone and at the water table, z_rz={kwave.z0}, zwt={kwave.zwt} cm",
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

    stats = pstats.Stats(pr)
    stats.sort_stats("cumtime").print_stats(20)
    
    # %%
