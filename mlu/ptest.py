'''Pumping test module.

Contains Ptest class that allows finite difference modeling of
axially symmetric flow in a multilayer confined aquifer system
and extracting the results at desired (piezometer) distances, layers
and times.

'''
tools = '/Users/Theo/GRWMODELS/python/tools/'

import os
import sys

if not tools in sys.path:
    sys.path.insert(1, tools)

import numpy as np
from fdm.mfgrid import Grid
from fdm.fdm3t  import fdm3t
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mlu.mlu_xml import Mluobj, mlu2xml
from colors import colors
from hantushn import hantushn

class Pumptest:

    def __init__(self, gr=None, kD=None, c=None, S=None, top_aquitard=False,
                       t=None, Q=None, obswells=None, **kwargs):
        '''Create a Pumptest object and simulate it with fdm and hantushn.

        'rw and R are assumed to be gr.x[0] and gr.x[-1]

        parameters
        ----------
            kD : sequence of floats
                transmissivities of aquifers [L2/T]
            S  : tuple (Saq, Sat) | sequence Saq
                Saq : squence of floats
                    storage coefficients of aquifers
                Sat : sequence of floats
                    storage coefficients of aquitards
                if not a tuplbe with two sequences of floats, then
                S is assumed Saq for aquifers with all Sat = 0
            c  : sequence of floats
                hydraulic resistances of aquitards [T]
            top_aquitard : bool
                whether top layer is an aquitard
            t : sequence of floats
                times to compute heads
            Q : sequence fo floats
                discharge from each aquifier
        '''

        gr.axial    = True
        self.t      = np.array(t)
        self.Q      = np.array(Q)
        self.gr     = gr; self.gr.Axial = True
        self.top_aquitard = top_aquitard

        assert self.gr.nz == len(kD) + len(c),\
            "len(kD) <{}> + len(c) <{}> != gr.nz <{}>"\
            .format(len(kD), len(c), gr.nz)
        if isinstance(S, tuple):
            assert len(kD) == len(S[0]),\
                'len(kD) <{}> != len(S[0]) <{}>'.format(len(kD), len(S[0]))
            assert len(c)  == len(S[1]),\
                'len(c) <{}> != len(S[[1]) <{}>'.format(len(c), len(S[1]))
        else: # if not a sequence, then S is Saq
            assert len(kD) == len(S),\
                "len(kD) <{}> != len(S) <{}>".format(len(kD), len(S))

        if len(c) > len(kD):
            top_aquitard = True

        #bot_aquitard = (len(c)  > len(kD)) or \
        #               (len(c) == len(kD)) and (not top_aquitard)

        self.iaquif = np.ones(self.gr.nz, dtype=bool)
        if self.top_aquitard:
            self.iaquif[0::2] = False
        else:
            self.iaquif[1::2] = False

        self.iatard = np.logical_not(self.iaquif)

        # conductivities
        kh = np.zeros(self.gr.nz)
        kh [self.iaquif] = np.array(kD) / self.gr.dz[self.iaquif]
        kv = np.ones(self.gr.nz) * 1e6 # just large
        kv [self.iatard] = self.gr.dz[self.iatard] / np.array(c)

        if self.iatard[ 0]: kv[ 0] /= 2.
        if self.iatard[-1]: kv[-1] /= 2.

        self.Kh = self.gr.const(kh)
        self.Kv = self.gr.const(kv)

        self.Kh[self.iaquif,:,0] = 100. # no resistance inside well

        # aquitards and aquifers can both have storage
        ss = np.zeros(self.gr.nz)
        if isinstance(S, tuple):
            Saq = S[0]
            Sat = S[1]
            assert len(Saq) == len(kD), 'len(Saq) <{}> != len(kD)<{}>'\
                .format(len(Saq), len(kD))
            assert len(Sat) == len(c), 'len(Sat) <{}> != len(kD) <{}>'\
                .format(len(Sat), len(kD))
            ss[self.iaquif] = Saq / self.gr.dz[self.iaquif]
            ss[self.iatard] = Sat / self.gr.dz[self.iatard]
        else: # S applies to only aquifers
            Saq = S
            Sat = np.zeros_like(c)
            assert len(Saq) == len(S), 'len(Saq) <{}> != len(kD) <{}>'\
                .format(len(Saq), len(kD))
            ss[self.iaquif] = S / self.gr.dz[self.iaquif]

        self.Ss = self.gr.const(ss)

        # Make sure t starts at zero to include initial heads.
        self.t = np.unique(np.hstack((0., np.array(t))))
        assert np.all(t)>=0, "All times must be > zero."

        # Discharge must be given for each aquifer
        assert len(Q) == len(kD), "len(Q) <{}> != len(kD) <{}>," +\
                    "specify a discharge for each aqufier."\
                    .format(len(Q), len(kD))

        # Boundary and initial conditions
        self.FQ     = self.gr.const(0)
        self.FQ[self.iaquif, 0, 0] = self.Q

        # Initial condition all heads zero (always in pumping test)
        self.HI     = self.gr.const(0.)

        # Boundary array (Modflow-like)
        self.IBOUND = self.gr.const(1, dtype=int)
        # Only prescribe heads to be fixed if aquitard on top and or bottom
        # Just make sure that the radius of the model is large enough.

        if self.iatard[ 0]: self.IBOUND[ 0] = -1
        if self.iatard[-1]: self.IBOUND[-1] = -1

        # Simulate using FDM
        self.out = fdm3t(gr=self.gr, t=self.t,
                         kxyz=(self.Kh, self.Kh, self.Kv),
                         Ss=self.Ss, FQ=self.FQ, HI=self.HI,
                         IBOUND=self.IBOUND, epsilon=1.0)

        print(' ')


        # Also compute Hantush. Hantush computes at observation distances.
        # For this we need the observation points
        self.names = list()
        self.r_ow  = list()
        self.z_ow  = list()
        assert obswells is not None, "obswells == None, specify obswells as [(name, r, z), ..]"
        for ow in obswells:
            self.names.append(   ow[0])
            self.r_ow.append(    ow[1])
            self.z_ow.append(ow[2])

        # store layer nr of each obswell
        self.layNr = self.gr.lrc(self.r_ow, np.zeros(len(self.r_ow)), self.z_ow)[:, 0]
        # get and store aquifer number of each obswell
        aqNr = np.zeros(self.gr.nz, dtype=int) - 1
        k = 0
        for i, ia in enumerate(self.iaquif):
            if ia:
                aqNr[i] = k
                k +=1
        self.aqNr = aqNr[self.layNr]
        assert np.all(self.aqNr) > -1, "one or more obswells not in aquifer"

        self.dd = hantushn(Q=self.Q, r=self.r_ow, t=self.t[1:],
                               Saq=Saq, Sat=Sat, c=c, T=kD, N=8)


    def show(self, **kwargs):
        '''Plot the t dd curves for the observation points.

        Plots both the numerical and analytical solutions.
        '''

        # Numerical solutions interpolated at observation points

        # Get itnepolator but also squeeze out axis 2 (y)
        interpolator = interp1d(self.gr.xm, self.out['Phi'][:, :, 0, :], axis=2)

        # interpolate at radius of observation points
        phi_t = interpolator(self.r_ow)

        # prepare fance selection of iz, ix combinations of obs points
        Ipnt = np.arange(phi_t.shape[-1], dtype=int) # nr of obs points

        phi_t = phi_t[:, self.layNr, Ipnt] # fancy selection, all times

        # Select by fancy indexing the obs point drawdowns from Hantush output
        ddHant = self.dd[:, self.aqNr, Ipnt]

        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title('Ptest testing')
            ax.set_xlabel('t [d]')
            ax.set_ylabel('drawdown [m]')
            ax.grid(True)
            ax.invert_yaxis()

        xscale = kwargs.pop('xscale', None)
        yscale = kwargs.pop('yscale', None)
        grid   = kwargs.pop('grid'  , None)

        if xscale: ax.set_xscale(xscale)
        if yscale: ax.set_yscale(yscale)
        if grid:   ax.grid(grid)

        # Numeric, fdm
        for fi, label, color in zip(phi_t.T, self.names, colors):
            ax.plot(self.t[1:], fi[1:], color=color, label=label + ' fdm', ls='', marker='x', **kwargs)

        # Analytic, Hanush
        for fi, label, color in zip(ddHant.T, self.names, colors):
            ax.plot(self.t[1:], fi, color=color, label=label + ' ana', ls='-', **kwargs)

        ax.legend(loc='best')

        return ax


class MLU_ptest:

    def __init__(self, mluxmlfile, tshift=0, Rmax=1e4, **kwargs):
        '''Return MLUtest, pumping test results based on mlu file.

        parameters
        ----------
            tshift : float
                Delay imposed on measurement time (pump upstart delay)
            Rmax: float
                Duter radius of model where head is fixed.

        kwargs
        ------
            Rw : float
                Well radius (not in mlu file) if None, use 2 * screen of well
        '''

        mlu = Mluobj(mluxmlfile)
        aqSys = mlu.aqSys
        self.obswells = mlu.obsWells

        nz = len(aqSys.aquifs) + len(aqSys.atards)
        self.iaquif = np.ones(nz, dtype=bool)
        self.top_aquitard = aqSys.top_aquitard != 'absent'
        if self.top_aquitard:
            self.iaquif[::2] = False
        else:
            self.iaquif[1::2] = False
        self.iatard = np.logical_not(self.iaquif)

        rmin   = kwargs.pop('Rw', 2 * mlu.wells[0].screen)

        # Piezometer data from xml file
        tmax   = 0
        tmin   = np.inf
        x0, y0 = mlu.wells[0].x, mlu.wells[0].y
        r      = rmin
        rmax   = rmin
        self.names    = list()
        self.points   = list()
        self.ow_aquif = list()

        for ow in mlu.obsWells:
            self.names.append(ow.name)
            r    = max(rmin, np.sqrt((x0 - ow.x)**2 + (y0 - ow.y)**2))
            rmax = max(rmax, r)
            tmin = min(tmin, ow.data[0, 0])
            tmax = max(tmax, ow.data[-1,0])
            self.ow_aquif.append(ow.layer - 1)
            self.points.append((r, mlu.aqSys.aquifs[ow.layer - 1].zmid))

        # attribute discharge over screened aquifers
        Qs    = np.zeros(len(mlu.aqSys.aquifs))
        kDtot = 0.
        for iaq, (screened, aq) in enumerate(zip(mlu.wells[0].screens, mlu.aqSys.aquifs)):
            if screened=='1':
                kDtot += aq.T
                Qs[iaq]  = aq.T
        Qs *= mlu.wells[0].Q / kDtot

        # radial grid, general max r.
        r  = np.hstack((0, np.logspace(np.log10(rmin), np.log10(Rmax),
                         int(10 * np.ceil(np.log10(Rmax/rmin)) + 1))))


        zf = np.array([(aquif.ztop, aquif.zbot) for aquif in aqSys.aquifs])
        za = np.array([(atard.ztop, atard.zbot) for atard in aqSys.atards])
        z = np.unique(np.hstack((zf[:,0], zf[:,1], za[:,0], za[:,1])))

        self.gr = Grid(r, [-0.5, 0.5], z, axial=True)

        # sufficiently detailed times for graphs
        self.t = np.logspace(np.log10(tmin), np.log10(tmax), 60)
        self.t = np.unique(np.hstack((0., self.t))) # start at 0

        # conductivities
        kD = np.array([aq.T for aq in aqSys.aquifs])
        c  = np.array([at.c for at in aqSys.atards])
        kh = np.zeros(self.gr.nz)
        kh [self.iaquif] = kD / self.gr.dz[self.iaquif]
        kv = np.ones(self.gr.nz) * 1e6 # just large
        kv [self.iatard] = self.gr.dz[self.iatard] / c

        if self.iatard[ 0]: kv[ 0] /= 2.
        if self.iatard[-1]: kv[-1] /= 2.

        self.Kh = self.gr.const(kh)
        self.Kv = self.gr.const(kv)

        self.Kh[self.iaquif,:,0] = 100. # no resistance inside well

        # storativities (both aquitards and aquifers)
        Saq = np.array([aq.S for aq in aqSys.aquifs])
        Sat = np.array([at.S for at in aqSys.atards])
        ss = np.zeros(self.gr.nz)
        ss[self.iaquif] = Saq / self.gr.dz[self.iaquif]
        ss[self.iatard] = Sat / self.gr.dz[self.iatard]

        self.Ss = self.gr.const(ss)

        kd_tot = 0.
        kd = np.zeros_like(kD)
        for i, scr in enumerate(mlu.wells[0].screens):
            if scr=='1':
                kd[i]   = kD[i]
                kd_tot += kD[i]
        self.Q = mlu.wells[0].Q * kd / kd_tot

        self.FQ     = self.gr.const(0)
        self.FQ[self.iaquif, 0, 0] = self.Q
        self.HI     = self.gr.const(0.)

        self.IBOUND = self.gr.const(1, dtype=int)
        self.IBOUND[:,:,-1] = -1

        if self.iatard[ 0]: self.IBOUND[ 0] = -1
        if self.iatard[-1]: self.IBOUND[-1] = -1

        # TODO: verify that the inflow over the outer boundary is < 5% or so.

        self.out = fdm3t(gr=self.gr, t=self.t, kxyz=(self.Kh, self.Kh, self.Kv),
                         Ss=self.Ss, FQ=self.FQ, HI=self.HI,
                         IBOUND=self.IBOUND, epsilon=1.0)


        ax = mlu.plot_drawdown(mlu.obsNames, tshift=tshift,
                               marker='.', linestyle='none', **kwargs)


        #hantush(...)
        r = np.array(self.points)[:, 0]
        self.dd = hantushn(Q=self.Q, r=r, t=self.t[1:],
                               Saq=Saq, Sat=Sat, c=c, T=kD, N  =8)

        self.show(xscale='log', ax=ax, labels=self.names)



    def show(self,**kwargs):

        #points should be r, z tuples
        rz = np.array(self.points)
        if rz.shape[1] == 2:
            rz = np.hstack((rz[:, 0:1], np.zeros((rz.shape[0], 1)), rz[:,1:2]))

        LRC = self.gr.lrc(rz[:,0], rz[:,1], rz[:,-1] )

        # squeeze out axis 2 (y)
        interpolator = interp1d(self.gr.xm, self.out['Phi'][:, :, 0, :], axis=2)

        # interpolate for all obswells at once along r-axis
        phi_t = interpolator(rz[:,0])

        # fancy indexing to select the correct col-row (r-z) combinations
        Ipnt = np.arange(phi_t.shape[-1], dtype=int)
        phi_t = phi_t[:, LRC[:,0], Ipnt]

        # Add Hantushn points
        aqNr =  np.array(self.ow_aquif)
        ddHant= self.dd[:, aqNr, np.arange(self.dd.shape[-1], dtype=int)]


        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title('Ptest testing')
            ax.set_xlabel('t [d]')
            ax.set_ylabel('drawdown [m]')
            ax.grid(True)
            ax.invert_yaxis()

        xscale = kwargs.pop('xscale', None)
        yscale = kwargs.pop('yscale', None)
        grid   = kwargs.pop('grid'  , None)

        if xscale: ax.set_xscale(xscale)
        if yscale: ax.set_yscale(yscale)
        if grid:   ax.grid(grid)

        labels = kwargs.pop('labels', ['']*phi_t.shape[1])

        # plot fdm3t lines
        for fi, label, color in zip(phi_t.T, labels, colors):
            ax.plot(self.t[1:], fi[1:], color=color, label=label + ' fdm', ls='--', marker='+', **kwargs)

        # plot hantush lines
        for fi, label, color in zip(ddHant.T, labels, colors):
            ax.plot(self.t[1:], fi, color=color, label=label + ' anal', lw=2,  **kwargs)

        ax.legend(loc='best')

        return ax


if __name__=='__main__':

    datapath = '/Users/Theo/GRWMODELS/python/tools/mlu/testdata/'

    ptestName  = 'gat_boomse_klei'
    #ptestName  = 'zuidelijke_landtong'

    mlu_file = os.path.join(datapath, ptestName + '.mlu')
    xml_file = os.path.join(datapath, ptestName + '.xml')

    mlu2xml(mlu_file, fout=xml_file)

    mptest = MLU_ptest(xml_file, Rw=0.13, tshift=0.)

    # =============gat_boomse_klei using ptest ================================

    # Firstly define the grid
    Rw   = 0.13    # m, well borehole radius
    Rmax = 10000. # m, extent of model (should be large enough)

    r  = np.hstack((0, np.logspace(np.log10(Rw), np.log10(Rmax),
                                   np.ceil(10 * np.log10(Rmax/Rw) + 1))))

    dz = np.array([9., 16., 0.01, 3.79, 3.7, 1., 3.7, 1., 3.8, 1., 1.3, 5., 3.4, 6.3])
    z  = np.hstack((0, -np.cumsum(dz)))
    gr = Grid(r, [-0.5, 0.5], z, axial=True)

    # Set soil parameters
    kD = np.array([48, 11.37, 0.13, 0.15, 0.16, 0.75, 31.5])
    c  = np.array([1500, 1.e-2, 2.85, 12.3, 76., 26., 56.7])
    Sat= np.array([0., 0., 0., 0., 0., 0., 0.,])
    Saq =np.array([1.e-04, 1.e-4, 1.e-5, 1.e-5, 1.e-5, 1.e-5, 1.e-5])
    top_aquitard = True  # whether the top layer is a top aquitard
    Q = np.array([0., 0., 0., 0., 0., 40.77, 0.])       # extraction from each aquifer

    # simulation time
    t = np.logspace(-5, 1, 101) # times for simulation
    # specify observation points [(anme, r, iaquifer), ...]
    well = ('pp4'   , 45619.637, 372414.25, 5)

    points = [('pb11_3', 45620.875, 372422.43, 2),
              ('pb11_4', 45622.344, 372423.27, 3),
              ('pb11-5', 45621.608, 372421.04, 4),
              ('pb11-6', 45619.263, 372421.36, 5),
              ('pp4'   , 45619.637, 372414.25, 5)]

    # Generate points as [(name, r, z), (...), ...]
    # Distance to well
    xy = np.array([(p[1], p[2]) for p in points])
    r_ow= np.sqrt((well[1] - xy[:,0])**2 + (well[2] - xy[:,1])**2)
    # Make sure min distance equals well radius
    r_ow = np.fmax(r_ow, Rw)

    # Convert (zero-based) aquifer numbers to z of aquifer center
    Iaq = np.array([p[3] for p in points], dtype=int) # aquifer number
    Ilay = Iaq * 2 + 1 if top_aquitard else Iaq * 2   # fdm layer number
    z_ow = (z[Ilay] + z[Ilay + 1]) / 2.                  # z of layer center

    # generate obswells for ptest
    obswells = [(p[0], r, zm_) for p, r, zm_ in zip(points, r_ow, z_ow)]


    # Get pumping test simulation object, it runs immediately
    pt = Pumptest( gr=gr, kD=kD, c=c, S=(Saq, Sat), top_aquitard=True,
                   t=t, Q=Q, obswells=obswells)

    # show the drawdown curves for the observation points
    pt.show(xscale='log', grid=True)

    # ============ setting up a the pumping test (not using mlu file) =========

    # Firstly define the grid
    Rw   = 0.25    # m, well borehole radius
    Rmax = 10000. # m, extent of model (should be large enough)

    r  = np.logspace(np.log10(Rw), np.log10(Rmax), np.ceil(10 * np.log10(Rmax/Rw) + 1))
    z  = [0, -4, -20, -22, -40]
    gr = Grid(r, [-0.5, 0.5], z, axial=True)

    # Set soil parameters
    kD = [200, 400]      # m2/d, transmissivities
    c  = [150, 500]      # d, hydraulic aquitard resistances
    S = [0.001, 0.001]   # [-], storage coefficients
    top_aquitard = True  # whether the top layer is a top aquitard
    Q = [100, 500]       # extraction from each aquifer

    # simulation time
    t = np.logspace(-3, 1, 101) # times for simulation

    # specify observation points [(anme, r, iaquifer), ...]
    points = [('wp0', 15, -5), ('wp1', 25, -10),
              ('wp2', 50, -15), ('wp3', 35, -25),
              ('wp4', 110, -30)]


    # Get pumping test simulation object, it runs immediately
    pt = Pumptest( gr=gr, kD=kD, c=c, S=S, top_aquitard=True,
                   t=t, Q=Q, obswells=points)

    # show the drawdown curves for the observation points
    pt.show(xscale='log', grid=True)

