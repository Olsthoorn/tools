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


class Pumptest:


    def __init__(self, gr=None, kD=None, c=None, S=None, topaquitard=False,
                       t=None,
                       Q=None, **kwargs):
        '''Create a Pumptest object
        kD : sequence of floats
            transmissivities of aquifers [L2/T]
        S  : squence of floats or a tuple with two sequences of floats
            if sequence of floats, then S is for aquifers
            if two sequences, the first if for aquifers, the second for aquiards
        c  : sequence of floats
            hydraulic resistances of aquitards [T]
        topaquitard : bool
            whether top layer is an aquitard
        t : sequence of floats
            times to compute heads
        Q : sequence fo floats
            discharge from each aquifier
        '''


        self.Q      = np.array(Q)
        self.gr     = gr; self.gr.Axial = True
        self.topaquitard = topaquitard

        assert self.gr.nz == len(kD) + len(c),\
            "len(kD) <{}> + len(c) <{}> != gr.nz <{}>"\
            .format(len(kD), len(c), gr.nz)
        if isinstance(S, (list, tuple)):
            assert len(kD) == len(S[0]),\
                'len(kD) <{}> != len(S[0]) <{}>'.format(len(kD), len(S[0]))
            assert len(c)  == len(S[1]),\
                'len(c) <{}> != len(S[[1]) <{}>'.format(len(c), len(S[1]))
        else:
            assert gr.nz == len(S),\
                "gr.nz <{}> != len(S) <{}>".format(gr.nz, len(S))

        iaquif = np.ones(self.gr.nz, dtype=bool)
        if self.topaquitard:
            iaquif[0::2] = False
        else:
            iaquif[1::2] = False
        iatard = iaquif == False

        k = np.zeros(self.gr.nz)
        k [iaquif] = np.array(kD) / self.gr.dz[iaquif]
        k [iatard] = self.gr.dz[iatard] / np.array(c)
        self.K = self.gr.const(k)

        # implies that aquiftards and aquifers can both have storage
        ss = np.zeros(self.gr.nz)
        if isinstance(S, (list, tuple)):
            ss[iaquif] = S[0] / self.gr.dz[iaquif]
            ss[iatard] = S[1] / self.gr.dz[iatard]
        else:
            ss[iaquif] = S

        self.Ss = self.gr.const(ss)

        self.iaquif = iaquif
        self.iatard = iatard

        if iatard[ 0]: self.K[ 0] = self.K[ 0] / 2.
        if iatard[-1]: self.K[-1] = self.K[-1] / 2.


        # make sure t starts at zero.
        self.t      = np.unique(np.hstack((0., np.array(t))))
        assert np.all(t)>=0, AssertionError("All times must be > zero.")

        assert len(Q) == len(kD), AssertionError("len(Q) <{}> != len(kD) <{}>," +
                   "specify a discharge for each aqufier."
                   .format(len(Q), len(kD)))

        self.FQ     = self.gr.const(0)
        self.FQ[iaquif, 0, 0] = self.Q
        self.HI     = self.gr.const(0.)

        self.IBOUND = self.gr.const(1, dtype=int)
        self.IBOUND[:,:,-1] = -1

        if self.iatard[0]:  self.IBOUND[0] =  -1
        if self.iatard[-1]: self.IBOUND[-1] = -1

        # TOTO: verify that the inflow over the outer boundary is < 5% or so.


    def run(self, **kwargs):
        '''Run the axial pumptest model, producing results

        kwargs
        ------
            t : ndarray(floats)
                time
            gr : mfgrid.Grid
                grid object holding the FD network
            K : ndarray [Nz, Ny, Nx] of floats
                hydraulic conductivity for all cells (vertically istropic)
            Ss : ndarray [Nz, Ny, Nx]
                Specific storage coeffiicents for all cells
            FQ : ndarray [Nz, Ny, Nx]
                Fixed flow for all cells
            HI : ndarray [Nz, Ny, Nx]
                Initial head for all cells
            IBOUND : ndarray [Nz, Ny, Nx] of dtype int
                Bounary array for all cells.
            epsilon : float
                implicitness (MODFLOW uses 1.0)
        returns
        -------
            Out, ordered dict with 3D arrays for
                Phi : ndarray [Nt + 1, Nz, Ny, Nx]
                    initial and computed heads
                Q   : ndarray  [Nt, Nz, Ny, Nx]
                    nodal flows during each stress period
                Qs  : ndarray [Nt, Nz, Ny, Nx]
                    storage during each stress period
                Qx : ndarray [ Ny, Nz, Ny, Nx - 1]
                    flows in each cell in x-direction
                Qy : ndarray [Nt, Nz, Ny- 1, Nx]
                    flows in each cell in y direction (0 in axially symmetric models)
                Qz : ndarray [Ny, Nz - 1, Ny, Nx]
                    flows in each cell in z direction
        '''

        # set conductivity of cells with extraction to large
        self.K[self.iaquif, 0, 0] =100.

        self.out = fdm3t(gr=kwargs.pop('gr',self.gr),
                         t=kwargs.pop( 't', self.t),
                         kxyz=kwargs.pop( 'K', self.K),
                         Ss=kwargs.pop('S', self.Ss),
                         FQ=kwargs.pop('FQ',self.FQ),
                         HI=kwargs.pop('HI',self.HI),
                         IBOUND=kwargs.pop('IBOUND',self.IBOUND),
                         epsilon=kwargs.pop('epsilon', 1.0))

        return self.out

    def show(self, points, **kwargs):

        #points should be r, z tuples
        rz = np.array(points)
        if rz.shape[1] == 2:
            rz = np.hstack((rz[:, 0:1], np.zeros((rz.shape[0], 1)), rz[:,1:2]))

        LRC = self.gr.lrc(rz[:,0], rz[:,1], rz[:,-1] )

        #import pdb
        #pdb.set_trace()

        interpolator = interp1d(self.gr.xm, self.out['Phi'], axis=3)
        phi_t = interpolator(rz[:,0])[:, :, 0, :]
        Ipnt = np.arange(phi_t.shape[-1], dtype=int)
        phi_t = phi_t[:, LRC[:,0], Ipnt]


        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title('Ptest testing')
            ax.set_xlabel('t [d]')
            ax.set_ylabel('drawdown [m]')
            ax.grid(True)

        xscale = kwargs.pop('xscale', None)
        yscale = kwargs.pop('yscale', None)
        grid   = kwargs.pop('grid'  , None)

        if xscale: ax.set_xscale(xscale)
        if yscale: ax.set_yscale(yscale)
        if grid:   ax.grid(grid)

        labels = kwargs.pop('labels', ['']*phi_t.shape[1])

        for fi, label, color in zip(phi_t.T, labels, colors):
            ax.plot(self.t[1:], fi[1:], color=color, label=label, **kwargs)

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

        rmin   = kwargs.pop('Rw', 2 * mlu.wells[0].screen)

        ax = mlu.plot_drawdown(mlu.obsNames, tshift=tshift,
                               marker='.', linestyle='none', **kwargs)

        self.kD =list()
        self.Saq = list()
        self.z = [mlu.aqSys.top_level]
        for aquif in mlu.aqSys.aquifs:
            self.kD.append(aquif.T)
            self.Saq.append(aquif.S)
            self.z.append(aquif.zbot)

        self.c = list()
        self.Sat = list()
        for atard in mlu.aqSys.atards:
            self.c.append(atard.c)
            self.Sat.append(atard.S)
            self.z.append(atard.zbot)

        top_aquitard= False if mlu.aqSys.top_aquitard == 'absent' else True

        tmax   = 0
        tmin   = np.inf
        x0, y0 = mlu.wells[0].x, mlu.wells[0].y
        r      = rmin
        rmax   = rmin
        names   = list()
        ow_points = []
        for ow in mlu.obsWells:
            names.append(ow.name + ' (simulated)')
            r = max(rmin, np.sqrt((x0 - ow.x)**2 + (y0 - ow.y)**2))
            rmax = max(rmax, r)
            tmin = min(tmin, ow.data[0, 0])
            tmax = max(tmax, ow.data[-1,0])
            ow_points.append((r, mlu.aqSys.aquifs[ow.layer - 1].zmid))

        #TODO: check of ow.layer is zero based

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
        y = np.array([-0.5, 0.5])
        z = np.array(sorted(self.z))

        self.gr = Grid(r, y, z, axial=True)

        ptest = Pumptest(self.gr, self.kD, self.c, (self.Saq, self.Sat), topaquitard=top_aquitard,
                       t=np.logspace(np.log10(tmin), np.log10(tmax), 60),
                       Q=Qs)

        self.out = ptest.run()

        ptest.show(ow_points, xscale='log', ax=ax, labels=names)


if __name__=='__main__':


    datapath = '/Users/Theo/GRWMODELS/python/tools/mlu/testdata/'


    '''Adaptation: storage coefficients of aquifers from 1e-5 to 1e-3
    '''
    #ptestName  = 'gat_boomse_klei_adapted'
    ptestName  = 'zuidelijke_landtong'

    mlu2xml(os.path.join(datapath, ptestName + '.mlu'),
            fout = os.path.join(datapath, ptestName + '.xml'))

    mptest = MLU_ptest(os.path.join(datapath,ptestName + '.xml'), Rw=0.13, tshift=0.)


    '''
    # setting for the pumping test
    Rw   = 0.25 # m, well borehole radius
    Rmax = 10000. # m, extent of model (should be large enough)

    # The fdm grid
    x  = np.logspace(np.log10(Rw), np.log10(Rmax), np.ceil(10 * np.log10(Rmax/Rw) + 1))
    z  = [0, -4, -20, -22, -40]
    gr = fdm.mfgrid.Grid(x, [-0.5, 0.5], z, axial=False)

    # parametes
    kD = [200, 400]     # m2/d, transmissivities
    c  = [150, 500]     # d, hydraulic aquitard resistances
    S = [0.001, 0.001]  # [-], storage coefficients
    topaquitard = True  # whether the top layer is a top aquitard
    t = np.logspace(-3, 1, 101) # times for simulation
    Q = [100, 500]      # extraction from each aquifer

    # Get pumping test object
    pt = Pumptest( gr=gr, kD=kD, c=c, S=S, topaquitard=True,
                       t=t,
                       Q=Q)

    # run the test (simulate)
    out = pt.run(t=t)

    # observation points [(r, z), ...]
    points = [(15, -5), (25, -10), (50, -15), (35, -25), (110, -30)]

    # show the drawdown curves for the observation points
    pt.show(points, xscale='log', grid=True)
    '''

