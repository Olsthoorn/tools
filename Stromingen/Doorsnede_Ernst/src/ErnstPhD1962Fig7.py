"""Study the breakup of the flow in fig7 in Ernst's PhD of 1962."""
# %%
# [tool.ruff]
# select = ["E", "F"]
# ignore = ["I001"]

import os
from pathlib import Path
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch, Path as PPath

from fdm.src import fdm3
from fdm.src.mfgrid import Grid


# %%
cwd = os.getcwd()
parts = Path(cwd).parts
assert '2022-AGT' in parts, "2022-AGT must be in path for correct saving: {home}"
images = os.path.join(*parts[:parts.index("2022-AGT") + 1], "Coding", "images")


def objpatch(xy, **kwargs):
    """Generate a patch given xy in an array."""
    verts = np.vstack([xy, xy[0]])
    codes = np.full(len(verts), PPath.LINETO)
    codes[0] = PPath.MOVETO
    codes[-1] = PPath.CLOSEPOLY

    path = PPath(verts, codes)
    patch = PathPatch(path, fill=True, **kwargs)
    return patch

# %% The freatic head with and without vertical

def xsection_7b():
    b, D, kh, kv, N = 100, 10, 5., 0.1, 0.001
    x = np.linspace(0, b, 1000)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Stroming in vert. sectie volgens Ernst (1952, fig 7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')

    x0s = np.linspace(0, b, 21)
    x0s[0] = 0.1

    for x0 in x0s:
        z0 = D
        z = x0 * z0 / x
        ax.plot(x[x>=x0], z[x>=x0])

    phi1 = D + N / (2 * kh * D) * (b ** 2 - x ** 2)
    phi2 = phi1 + N * D / (2 * kv) * (1 - (x/b) ** 2)

    ax.plot(x, phi1, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2, label="Freatisch vlak zonder verticale weerstand")
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(images, "Ernst_Xsec_basic"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Freatisch vlak met en zonder verticale weerstand naar Ernst (1952, fig 7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')
    ax.plot(x, phi1 - D, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2 - D, label="Freatisch vlak zonder verticale weerstand")

    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(images, "Ernst_Xsec_freatisch"))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Freatisch vlak met en zonder verticale weerstand naar Ernst (1962, fig.7), vaste D"
                "\n"
                f"D={D} m, b={b} m, N={N} m/d, kh={kh} m/d, kv={kv} m/d")
    ax.set(xlabel='x [m]', ylabel='z [m]')
    ax.plot(x, phi1 - D, label="Freatisch vlak zonder verticale weerstand")
    ax.plot(x, phi2 - D, label="Freatisch vlak zonder verticale weerstand")

    ax.grid(True)
    ax.legend()
    fig.savefig(os.path.join(images, "Ernst_Xsec_freatisch"))

    plt.show()

def get_Z(b, D, N=100, clip=1e-3):
    x = np.linspace(0, b, int(b/D * N + 1))
    y = np.linspace(0, D, N+ 1)
    X, Y = np.meshgrid(x, y)
    Z = X.clip(clip, None) + 1j * (Y.clip(clip, D-clip))
    return Z

def stroming_analytisch(b=40, D=10, dxy=0.1, N=0.001, k=1, case=None, ax=None):    
    kh, kv = k, k
    Q = N * b
    dQ = Q / 20

    Z = get_Z(b, D, clip=1e-3)
    
    # Omega = N / (2 * D) * ((b**2 - Z.real**2) +  Z.imag**2  + 1j * Z.real * Z.imag)

    # --- Potentiaal vertical + horizonaw flow
    # --- Horizontal head loss (no contraction of stream lines)
    Phi1 = N / (2*D) * (b**2 - Z.real**2)
    # --- Vertical head loss (no contraction of stream lines)
    Phi2 = N/(2*D) * Z.imag**2
    # --- Total head loss (no contraction of stream lines)
    Phi = Phi1 + Phi2

    # --- Stream function
    # --- Psi = N/D * xy
    Psi = N/D  * Z.real * Z.imag
    
    # -- Complex potential
    Omega0 = Phi - 1j * Psi

    # --- Extraction, extraction at point (0, iD)
    Omega1 = Q/ (np.pi / 2) * np.log(np.sin(1j *np.pi/2 * ((b-Z)/D - 1j)))
    
    # --- Uniform flow to cancel the flow at x=b due to Omega1
    Omega2 =  -Q * ((b - Z)/D - 1j )

    if False:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title("Phi at different depths")
        for z, omega in zip(Z, Omega0):
            ax.plot(z.real, omega.real, label=f"x={z[0].imag}")


    fig, axs = plt.subplots(3, sharex=True, sharey=True, figsize=(8, 11))
    fig.suptitle("Analytische doorrekening. doorsnede Ernst (1962, fig.7)"
                 "\n"
                 fr"Blauw: $\phi$, $d\phi$={dQ/k:.3f} m. Rood: $\Psi$, $d\Psi$={dQ:.5f} m2/d"
                 "\n"
                 f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, Q=Nb={N * b} m2/d")

    titles = ["Combinatie van Ernst (1962), fig.7b en fig.7c",
              "Stroming door contractie van stroomlijnen (Ernst (1962), fig.7d)",
              "Samengestelde stroming in de doorsnede (Ernst (1962), fig.7a)"]
        
    for ax, title in zip(axs, titles):
        ax.set_title(title)
        ax.set(xlabel='x[m]', ylabel='y[m]')
        if ax == axs[0]:
            Omega = Omega0
        elif ax == axs[1]:
            Omega = Omega1 + Omega2
        elif ax == axs[2]:
            Omega = Omega0 + (Omega1 + Omega2)
        
        # --- Phi and psi levels:
        phimin = np.floor(Omega.real.min() * 100)/100 / k
        phimax = np.ceil(Omega.real.max() * 100)/100 / k
        psimin = np.floor(Omega.imag.min() * 100)/100
        psimax = np.ceil(Omega.imag.max() * 100)/100
        
        phiLevels = np.arange(phimin, phimax, dQ/k)
        psiLevels = np.arange(psimin, psimax, dQ)

        Cphi = ax.contour(Z.real, Z.imag, Omega.real/k, levels=phiLevels,
                   colors='b',
                   linestyles='solid',
                   linewidths=0.5)
        # ax.clabel(Cphi, levels=phiLevels)
        
        Cpsi = ax.contour(Z.real, Z.imag, Omega.imag, levels=psiLevels,
                   colors='r',
                   linestyles='solid',
                   linewidths=0.5)
        # ax.clabel(Cpsi, levels=psiLevels)
        ax.set_aspect(1)
        ax.set_ylim(0, 10)
        if not ax==axs[0]:
            ax.plot(b, D, 'ro', ms=20, mec='b', mfc='blue', zorder=100)

        fig.savefig(os.path.join(images, "Ernst_fig7_analytisch.png"))           

def xsec_Ernst_rectangular_case(b=20, D=10, dxy=0.1, N=0.01, k=1., case=5, ax=None):
    """Ernst's 1962 cross section is simulated, keeping the
    section thickness constant and reducting the ditch to a single
    cell of size dxy in the top right corner.
    
    The simulation is done by fdm3.py a steady state FDM model of my own making
    and available on https://github.com/Olsthoorn/tools/tree/master/fdm/src
    """
    Q = N * b
    dQ = Q / 20 # Used for distance between contours of head and stream function
    
    # --- Define the grid
    n = int(b / dxy) + 1
    m = int(D / dxy) + 1
    x = np.linspace(0, b, n+1)
    z = np.linspace(0, -D, m+1)
    gr = Grid(x, None, z, axial=False)

    # --- Define full size model arrays
    K = gr.const(k), gr.const(k), gr.const(k)
    FQ = gr.const(0.) # --- Fixed flows for all cells
    HI = gr.const(0)  # --- Initial heads

    IBOUND = gr.const(1, dtype=int)
    
    # --- possibly create axis or use given one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    # -- Set fig axis labels in axes-grid
    if case in [0, 2, 4]:
        ax.set_ylabel('z[m]')
    if case in [3, 4]:
        ax.set(xlabel='x[m]')

    if case==0:
        # --- Composite flow
        ax.set_title("Ernst fig 7a")
        FQ[0, 0, :] = N * gr.dx
        IBOUND[0, 0, -1] = -1
        
    elif case==1:
        # --- Only vertical flow
        ax.set_title("Ernst fig 7b")
        FQ = -N/D * gr.DX * gr.DZ
        FQ[0, 0, :] += N * gr.dx    
        IBOUND[-1, 0, -1] = -1
        
    elif case==2:
        # --- Only horizontal flow
        ax.set_title("Ernst fig 7c")
        FQ = +N/D * gr.DX * gr.DZ
        FQ[:, 0, -1] -= N *b / D * gr.dz    
        IBOUND[-1, 0, -1] = -1    
        
    elif case==3:
        # --- Combined vertical and horizontal flow
        ax.set_title("Ernst fig 7b + 7c")
        FQ = -N/D * gr.DX * gr.DZ
        FQ[0, 0, :] += N * gr.dx    
        FQ += N/D * gr.DX * gr.DZ
        FQ[:, 0, -1] -= N *b / D * gr.dz
        IBOUND[-1, 0, -1] = -1
        
    elif case==4:
        # --- Flow caused by partial penetration of ditch
        ax.set_title("Ernst fig 7d")
        FQ[:, 0, -1] = N * b / D * gr.dz
        IBOUND[0, 0, -1] = -1
        
    elif case==5:
        # --- Same as case 1, i.e. case 2 + case 3 + case 4
        ax.set_title("Ernst fig 7a")
        FQ[0, 0, :] = N * gr.dx
        IBOUND[0, 0, -1] = -1
        
    # --- Cell with ditch (which is reduced to only one cell)
    Id = gr.NOD[IBOUND == -1]

    # --- Plot this "ditch" of point with given zero head.
    if case in [0, 4, 5]:
        ms = 10
        ax.plot(gr.XM.ravel()[Id], gr.ZM.ravel()[Id], 'ro', ms=ms, mec='b', mfc='blue', zorder=100)
    
    # --- Simulate (compute heads and flows)
    out = fdm3.fdm3(gr, K=K, c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)
    
    print(f"phiMn={out['Phi'].min():.4g}, phiMax={out['Phi'].max():.4g}")
    phiLevels = np.arange(out['Phi'].min(), out['Phi'].max() + dQ, dQ / k)
            
    Cf = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], out['Phi'][:, 0, :],  levels=phiLevels,
                    colors='b',
                    linewidths=0.5,
                    linestyles='solid')
    # ax.clabel(Cf, levels=Cf.levels)
    
    if case not in [1, 2]:
        # --- Done compute and show streamlines in the cases with spatial divergence
        # --- because in those cases the stream function is not defined.
        
        sf = fdm3.psi(out['Qx']) # --- Stream function
        print(f"psiMin={sf.min():4g}, psiMax={sf.max():.4g}")        
        psiLevels = np.arange(sf.min(), sf.max() + dQ, dQ)
        
        Cs = ax.contour(gr.X[:, 0, 1:-1], gr.Z[:, 0, 1:], sf, levels=psiLevels,
                        colors='r',
                        linewidths=0.5,
                        linestyles='solid')
        # ax.clabel(Cs, levels=Cs.levels)

    ax.set_aspect(1)
    ax.set_xlim(b, 0)
    
    def w(z, D):
        """Return ditch resistance, one-sided flow.
        
        For two sided flow use have this value
        
        Parameters
        ----------
        z: complex coordinate
            The well is in z=0, and x>=0 and -D<y<0.
        """
        return 2 / np.pi * np.log(2 * np.sin(1j * np.pi * z / (2 * D))) - z/D
    
    if case==4:
        # Eenzijdige toestroming
        om1 = w(gr.xm, D)
        om2 = w(gr.xm - 1j * D, D)
        fig, ax = plt.subplots(figsize=(10,6))
        ax.set_title('Stijghoogte langs top en basis en benadering,  contractiestroming')
        ax.set(xlabel='x/D', ylabel='\phi [m]', xscale='log')
        ax.grid()
        ax.plot((b - gr.xm[:-1]) / D, out['Phi'][0, 0, :-1] - out['Phi'][0, 0, 0], 'b.', label=r'$\phi$ top, numeriek')
        ax.plot((b - gr.xm) / D, out['Phi'][-1, 0] - out['Phi'][-1, 0, 0], 'bx', label=r'$\phi$ basis, numeriek')
        ax.plot(gr.xm/D, Q * om1.real / k, 'g-', label=r'$\phi$ top, analytisch')
        ax.plot(gr.xm/D, Q * om2.real / k, 'g--', label=r'$\phi$ basis, analytisch')
        ax.plot(gr.xm/D, 2 * Q / (np.pi * k) * np.log(np.pi * gr.xm/D), 'r', label=r'$\frac{2Q}{\pi k}\,\ln\left(\frac{\pi x}{D}\right)$')
        # ax.plot(gr.xm/D, 2 * Q / (np.pi * k) * np.log(np.pi * gr.xm/D) - Q/D * gr.xm, 'c', label=r'$\frac{2Q}{\pi k}\,\ln\left(\frac{\pi x}{D}\right) - \frac{Q}{D} x$')
        # --- Ernst ditch (r = 2 m)
        r = 2
        ir = np.argmin(np.abs((b - gr.xm) - r))
        dphi_appr=2 * Q / (np.pi * k) * np.log(np.pi * r/D)        
        dphi_num = out['Phi'][0, 0, ir] - out['Phi'][0, 0, 0] 
        ax.plot(r/D, dphi_appr, 'o', ms=8, mfc='none', mec='r', label=f'sloot Ernst r={r}')
        ax.plot(r/D, dphi_num,  'o', ms=8, mfc='none', mec='b', label=f' sloot Ernst r={r}')
        
        
        
        ax.legend(loc='lower right')
        fig.savefig(os.path.join(images, "phi_top_basis.png"))
        
        
        # --- Fig. 11 Enrnst, tweezijdige toestroming
        fig, ax = plt.subplots(figsize=(6,6))

        ax.set_title('Fig 11 Ernst')
        ax.set(xlabel=r'$\pi r_0/D$', ylabel=r'$k w$ [m]', xscale='log')
                
        # --- w from om for Ernst: Q/2
        # ax.plot(np.pi * gr.xm/D, -om1.real / 2, 'b.', label=r'$\Phi/Q=kw$ top, analytisch')
        
        # --- w exactly the same, written out
        ax.plot(np.pi * gr.xm/D, -1/np.pi * np.log(2 * 1j * np.sinh(np.pi * gr.xm / (2 * D))) + gr.xm / (2 * D), 'g-',
                label=r'$-\frac{1}{\pi}\,\ln \left(2 i \sinh\left(\frac{\pi x}{2 D}\right)\right) + \frac{x}{2 D}$ (hor. top)')
        
        # --- w using sinh only
        ax.plot(np.pi * gr.xm/D, -1/np.pi * np.log(2 * np.sinh(np.pi * gr.xm / (2 * D))), 'r-', mfc='none', mec='g',
                label=r'$-\frac{1}{\pi}\,\ln \left(2\sinh\left(\frac{\pi x}{2 D}\right)\right)$')
        
        # --- w Huismand/Ernst
        ax.plot(np.pi * gr.xm/D, -1/np.pi * np.log(np.pi * gr.xm / D), 'r--', label=r'$-\frac{1}{\pi}\,\ln\left(\frac{\pi x}{D}\right)$')
        
        # --- w along vertical using Q / 2
        ax.plot(np.pi * gr.zm/D, -w(1j * gr.zm, D) / 2, 'b-',
                label=r'$-\frac{1}{\pi}\,\ln\left(2 \sin\left(\frac{\pi y}{2 D}\right)\right) + i \frac{y}{D}$ (vert. onder sloot)')
        
        # --- kw = 2r0/D  
        ax.plot(np.pi * gr.xm/D, +2*gr.xm/D, 'k--', label=r'$k w=\frac{2r_0}{D}$')
        
        ax.set_xlim(0.1, 3)
        ax.set_ylim(-0.8, 0.8)
        ax.grid(which='both')
        ax.legend(loc='lower left')
        
        fig.savefig(os.path.join(images, "Ernst_fig_11.png"))
        
        
    
    print(f"Done case {ic}")
    # --- Add field to out to use outside this function
    out['gr'] = gr
    return out

def xsec_Ernst_watertable_case(b=25, D=10, dxy=0.1, N=0.01, k=1., case=5, ax=None):
    """Ernst's 1962 cross section is simulated, using the actual water table
    as top of the flow domain and an actual ditch as the ditch. The relative
    sizes are closely the same as those of Ernst's figure 7.
    
    The simulation is done by fdm3.py a steady state FDM model of my own making
    and available on https://github.com/Olsthoorn/tools/tree/master/fdm/src
    """

    Q = N * b
    dQ = Q / 20 # --- dQ is used to set levels for contouring heads and streamlines
    
    # --- Define the grid
    n = int(b / dxy)
    m = int(1.2 * D / dxy)  # --- Grid reaches above zero to accommodate the higher water table
    x = np.linspace(0, b, n+1)
    z = np.linspace(0, 1.2 * D, m+1) - D
    gr = Grid(x, None, z, axial=False)

    # --- Grid size arrays
    K = gr.const(k)
    FQ = gr.const(0.) # --- Given flows
    HI = gr.const(0)  # --- Initial heads

    IBOUND = gr.const(1, dtype=int)
    
    # --- Ditch contour (its cross section coordinates)
    xy_ditch = np.array([(b-2, 0), (b-0.6, -2), (b, -2), (b, 0), (b-2, 0)])
    # --- Make it a patch to show it
    ditch = objpatch(xy_ditch, fc='blue', zorder=100)
    # --- Get cross section cells falling inside the ditch contour
    In = gr.inpoly(xy_ditch, row=0)
    # --- Make these cells info fixed head cells
    IBOUND[:, 0, :][In] = -1

    # --- Set axes labels for axes grid
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    if case in [0, 2, 4]:
        ax.set_ylabel('z[m]')
    if case in [3, 4]:
        ax.set(xlabel='x[m]')

    # --- get the phreatic surface: This is done by computing the heads several times and
    # --- each time making the cells above the water table (their bottom lower than the head)
    # --- inactive. The conductivities are also adapted to their degree of filling.
    cols = np.arange(gr.nx)   # --- Model grid column indices
    FQ[0, 0, : ] = N * gr.dx
    NIb = np.sum(IBOUND != 0) # ---- Just count how many cells are active

    for _ in range(10):
        # --- Simulate heads and flows
        out = fdm3.fdm3(gr, K=K, c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None)

        # --- determine wet cells
        fr = (out['Phi'] - gr.Z[1:]) / gr.DZ
        fr[fr > 1] = 1
        fr[fr < 0] = 0
        # --- adapt k to wetting of cells
        K = (fr * k).clip(1e-3)

        # --- make dry cells inactive
        IBOUND[fr == 0] = 0        
        mask = IBOUND != 0
        
        # --- Count the number of inactive cells
        NIb1 = np.sum(mask)
        print(f" {NIb1}", end=None)
        if NIb == NIb1:
            # --- number of inactive cells is stable, so quit
            print()
            break
        else:
            # --- Prepare for next iteration
            NIb = NIb1
            FQ = gr.const(0.)
            # --- get topmost active cell index for each columns
            k0 = mask.argmax(axis=0)     # k0[~mask.any(axis=0)] = -1
            # --- inject the recharge for the next iteration in these topmost active cells        
            FQ[k0, 0, cols] = N * dxy
    
    # --- Dompute the active length of each column (Water table minus aquifer base)        
    Dcols = (gr.DZ * (IBOUND==1)).sum(axis=0)[0]
    
    # --- Divergence to use in each column
    N_D = (N / Dcols) # N/D for each column (row vector)
    
    # --- Get the water table of the composite flow case (and keep it for all cases)
    hf = out['Phi'][k0, 0, cols]
    # --- Turn the area above the water table into a patch to cover that part of the Xsection
    xy_top = np.vstack((np.hstack((gr.xm, gr.xm[::-1], gr.xm[0])),
                        np.hstack((hf[0], gr.Z[0, 0, :][::-1], hf[0, 0]))
    ))
    top = objpatch(xy_top.T, fc='white', zorder=100)
    
    # --- Prepare and proceed with the actual cases (note that K and IBOUND remain the same for each case).
    FQ = gr.const(0.)

    if case==0:
        # --- composite flow (original)
        ax.set_title("Ernst fig 7a")
        # --- Only injection across the phreatic level
        FQ[k0, 0, cols] = N * dxy
          
    elif case==1:
        # --- only vertical flow
        ax.set_title("Ernst fig 7b")
        # --- injection at phreatic level
        FQ[k0, 0, cols] = N * gr.dx
        # --- (almost) uniform extraction
        FQ -= N_D[None, None, :] * gr.DX * gr.DZ
            
    elif case==2:
        # --- only horizontal flow
        ax.set_title("Ernst fig 7c")
        # --- (almost) uniform infiltration
        FQ = N_D[None, None, :] * gr.DX * gr.DZ
        # --- extraction at x=b below ditch
        FQ[IBOUND[:, 0, -1]==1, 0, -1] -= N*b / Dcols[-1] * gr.dz[IBOUND[:, 0, -1]==1]
        
    elif case==3:
        # --- flow combined without contraction flow lines
        ax.set_title("Ernst fig 7b + 7c")
        # --- injection at phreatic level
        FQ[k0, 0, cols] = N * gr.dx 
        # --- extraction at the right below the ditch       
        FQ[IBOUND[:, 0, -1]==1, 0, -1] -= N*b / Dcols[-1] * gr.dz[IBOUND[:, 0, -1]==1]

    elif case==4:
        # --- effect contracting flowlines
        ax.set_title("Ernst fig 7d")
        # --- injection total Nb below the ditch
        FQ[IBOUND[:, 0, -1]==1, 0, -1] += N*b / Dcols[-1] * gr.dz[IBOUND[:, 0, -1]==1]

    elif case==5:
        # --- original same as case 0
        ax.set_title("Ernst fig 7a")
        # --- Only injection across the phreatic level
        FQ[k0, 0, cols] = N * dxy        
        
    # --- Cover parts of the section to show ditch and don't show
    # --- stream vertical lines above the water table.
    # --- as well as do show the water table itself.
    ax.add_patch(ditch)
    ax.add_patch(top)

    # --- Simulate the current case
    out = fdm3.fdm3(gr, K=K, c=None, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=None) 
    
    # --- Compute head levels using dQ / k (dQ=PhiLevels and Psi levels, dQ/k is phiLevels)  
    phiLevels = np.arange(np.nanmin(out['Phi']), np.nanmax(out['Phi']), dQ/k)
            
    Cf = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], out['Phi'][:, 0, :],  levels=phiLevels,
                    colors='b',
                    linewidths=0.5,
                    linestyles='solid')
    #ax.clabel(Cf, levels=Cf.levels)
    
    if case in [1]:
        # --- Show Ernst's famous dashed line in his fig 7.2
        Cf = ax.contour(gr.XM[:, 0, :], gr.ZM[:, 0, :], out['Phi'][:, 0, :],  levels=[0.],
                    colors='b',
                    linewidths=2.0,
                    linestyles='dashed')
        ax.clabel(Cf, levels=[0.], fmt="%.2f")
        
    # --- Don't compute and plot stream lines for the cases with divergence
    # --- This is because the streamlines are not defined with divergence.
    if case not in [1, 2]:            
        print(f"phiMn={out['Phi'].min():.4g}, phiMax={out['Phi'].max():.4g}")        
        sf = fdm3.psi(out['Qx'])
        print(f"psiMin={sf.min():4g}, psiMax={sf.max():.4g}")        
        psiLevels = np.arange(sf.min(), sf.max(), dQ)        
        
        Cs = ax.contour(gr.X[:, 0, 1:-1], gr.Z[:, 0, 1:], sf, levels=psiLevels,
                        colors='r',
                        linewidths=0.5,
                        linestyles='solid')
        #ax.clabel(Cs, levels=Cs.levels)
    
    ax.set_aspect(1)
    
    ax.set_xlim(b, 0)
    print("Done case {ic}")
    # --- Add these fields to use outside this function
    out['gr'] = gr
    out['k0'] = k0[0]
    out['cols'] = cols
    return out

def partial_penetraton(b=25, D=10, N=None, k=None, dxy=0.1):
    Q = N * b
    dQ = Q / 20

    # --- With the top of the aquifer at y=0 an dthe bottom at y=-D
    Z = get_Z(b, D, clip=1e-6); Z.imag = -Z.imag
    
    # --- Extraction, extraction at point (0, iD)
    Omega = Q/ (np.pi / 2) * np.log(np.sin(1j *np.pi/2 * Z/D)) - Q * Z/D + 2*Q/np.pi * np.log(2)
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    fig.suptitle("Analytisch partial penetration"
                 "\n"
                 fr"Blauw: $\phi$, $d\phi$={dQ/k:.3f} m. Rood: $\Psi$, $d\Psi$={dQ:.5f} m2/d"
                 "\n"
                 f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, Q=Nb={N * b} m2/d")

    title="Stroming naar onttrekking in half oneindige doorsnede"
    xlabel="x [m]"
    ylabel="y [m]"
            
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    # --- Phi and psi levels:
    phimin = np.floor(Omega.real.min() * 100)/100 / k
    phimax = np.ceil(Omega.real.max() * 100)/100 / k
    psimin = np.floor(Omega.imag.min() * 100)/100
    psimax = np.ceil(Omega.imag.max() * 100)/100
    
    phiLevels = np.arange(phimin, phimax, dQ/k)
    psiLevels = np.arange(psimin, psimax, dQ)

    Cphi = ax.contour(Z.real, Z.imag, Omega.real/k, levels=phiLevels,
            colors='b',
            linestyles='solid',
            linewidths=0.5)
    # ax.clabel(Cphi, levels=phiLevels)
    
    Cpsi = ax.contour(Z.real, Z.imag, Omega.imag, levels=psiLevels,
            colors='r',
            linestyles='solid',
            linewidths=0.5)
    # ax.clabel(Cpsi, levels=psiLevels)
    
    ax.set_aspect(1)
    # ax.set(xlim=(0, 4), ylim=(-4, 0))
    
    fig.savefig(os.path.join(images, "Partial_penetration_analytic.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Stijghoogte langs de top en basis van de aquifer\n"
                 f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, Q=Nb={N * b} m2/d")
    ax.set(xlabel='x[m]', ylabel='phi[m]', xscale='log')
    x = Z[0].real
    ax.plot(x, Omega[0,  :].real/k, 'b-', label="horizontal, top")
    ax.plot(x, Omega[-1, :].real/k, 'b--', label="horizontal, base")
    # ax.plot(x, -2 * Q / np.pi * np.log(2) / k * np.ones(len(x)), 'g--', label=r'asymptoot $-(2Q/pi) \ln(2)$')
    
    r = x
    # ax.plot(r, 2 * Q / (np.pi * k) * np.log(np.pi * r / (2 * D)) * np.ones(len(r)), '-', label=r'Benadering $2(Q/\pi)\,\ln(\pi r/(2D))$')
    ax.plot(r, 2 * Q / (np.pi * k) * np.log(np.pi * r / D) * np.ones(len(r)), 'r-', label=r'Huisman $2(Q/\pi)\,\ln(\pi r/(D))$')
    ax.grid(True)
    ax.set_xlim(1e-2, None)
    ax.set_ylim(-0.15, 0.03)
    ax.legend()
    fig.savefig(os.path.join(images, "stijgh_top_basis_contractie.png"))
    
def partial_penenetration_anisotropie(b=25, D=10, N=None, kh=None, kv=None, dxy=0.1):
    """Handle partial ditch penteration for the anisotropic real world cross section.

    We use Z to represent the real-world anisotropic X-section.
    We use Ziso to represent the transformed isotropic X-section
    Likewise: biso, Diso represent the tranformed isotropic values.
    """
    Q = N * b
    dQ = Q / 20
    k = np.sqrt(kh*kv)
    biso, Diso = b * np.sqrt(k/kh), D * np.sqrt(k/kv)

    # --- With the top of the aquifer at y=0 an dthe bottom at y=-D
    Ziso = get_Z(b * np.sqrt(k/kh), D * np.sqrt(k/kv), clip=1e-6); Ziso.imag = -Ziso.imag
    Z = Ziso.copy()
    Z.real *= np.sqrt(kh/k)
    Z.imag *= np.sqrt(kv/k)
    
    def get_omega_iso(Q, Z, D):
        """Return Omega in isotropic medium with, extraction at point (0, iD)."""        
        Omega = Q/ (np.pi / 2) * np.log(np.sin(1j *np.pi/2 * Z/D)) - Q * Z/D + 2*Q/np.pi * np.log(2)
        return Omega
        
    Omega = get_omega_iso(Q, Ziso, Diso)
        
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8.5))
    
    fig.suptitle("Analytisch partial penetration"
                 "\n"
                 fr"Blauw: $\phi$, $d\phi$={dQ/k:.3f} m. Rood: $\Psi$, $d\Psi$={dQ:.5f} m2/d"
                 "\n"
                 f"b={b} m, D={D} m, N={N} m/d, kh={kh:.1f} m/d, kv={kv:.1f} m/d, Q=Nb={N * b} m2/d")
            
    ax1.set(title="Contractiestroming isotrope doorsnede", xlabel='x_acc [m]', ylabel='y_acc [m]')
    ax2.set(title="Contractiestroming anisotrope doorsnede", xlabel='x [m]', ylabel='y [m]')

    # --- Phi and psi levels:
    phimin = np.floor(Omega.real.min() * 100)/100 / k
    phimax = np.ceil(Omega.real.max() * 100)/100 / k
    psimin = np.floor(Omega.imag.min() * 100)/100
    psimax = np.ceil(Omega.imag.max() * 100)/100
    
    phiLevels = np.arange(phimin, phimax, dQ/k)
    psiLevels = np.arange(psimin, psimax, dQ)
    
    # --- Contour the transformed isotropic cross section
    Cphi = ax1.contour(Ziso.real, Ziso.imag, Omega.real/k, levels=phiLevels,
            colors='b',
            linestyles='solid',
            linewidths=0.5)
    # ax1.clabel(Cphi, levels=phiLevels)
    
    Cpsi = ax1.contour(Ziso.real, Ziso.imag, Omega.imag, levels=psiLevels,
            colors='r',
            linestyles='solid',
            linewidths=0.5)
    # ax1.clabel(Cpsi, levels=psiLevels)
    
    ax1.set_aspect(1)
    
    # --- Contour the anisotropic real-world cross section.
    Cphi = ax2.contour(Z.real, Z.imag, Omega.real/k, levels=phiLevels,
            colors='b',
            linestyles='solid',
            linewidths=0.5)
    # ax2.clabel(Cphi, levels=phiLevels)
    
    Cpsi = ax2.contour(Z.real, Z.imag, Omega.imag, levels=psiLevels,
            colors='r',
            linestyles='solid',
            linewidths=0.5)
    # ax2.clabel(Cpsi, levels=psiLevels)
    
    ax2.set_aspect(1)

    # --- Save the figure with both the isotropic and anisotropic X-sections.
    fig.savefig(os.path.join(images, "Partial_penetration_analytic_aniso.png"))

    # --- Proceed with the head along the top and bottom of the aquifer
    #     generated by the contraction of stream lines only.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Stijghoogte langs de top en basis van de aquifer\n"
                 f"b={b} m, D={D} m, N={N} m/d, kh={kh:.1f}, kv={kv:.1f} m/d, Q=Nb={N * b} m2/d")
    ax.set(xlabel='x/D', ylabel='phi[m]', xscale='log')
    x = Z[0].real
    ax.plot(x / D, Omega[0,  :].real/k, 'b-', label="horizontal, top")
    ax.plot(x / D, Omega[-1, :].real/k, 'b--', label="horizontal, base")
        
    r = x
    r_iso = Ziso[0].real
    ax.plot(r/ D, 2 * Q / (np.pi * k) * np.log(np.pi * r / D * np.sqrt(kv/kh)) * np.ones(len(r)), 'r-',
            label=r'Huisman $2(Q/\pi)\,\ln(\pi r/(D))$')
    #ax.plot(r/ D, 2 * Q / (np.pi * k) * np.log(np.pi * r_iso / Diso) * np.ones(len(r)), 'r.',
    #        label=r'Huisman $2(Q/\pi)\,\ln(\pi r/(D))$')
    
    ax.grid(True)
    ax.set_xlim(1e-2, None)
    ax.set_ylim(-0.02, None)
    ax.legend(loc='lower right')
    fig.savefig(os.path.join(images, "stijgh_top_basis_contractie_aniso.png"))

def comlexe_potentiaal(b=25, D=10, Q=1, d=1):
    n, m = int(b/d) + 1, int(D/d) + 2
    x = np.linspace(0, b, n); x[0] = 0.1
    y = np.linspace(0, -D, m).clip(-D + 1e-6, 0 - 1e-6)
    X,Y = np.meshgrid(x, y)
    Z = X + 1j * Y
        
    zta1 = 1j * np.pi/2 / D * Z
    zta2 = np.sin(zta1)
    zta3 = np.log(zta2)
    
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 7))
    ax1, ax2 = axs
            
    dQ = Q / 20
    
    # --- omega0
    omega = Q / (np.pi/2) * zta3
    
    phiLevels = np.arange(np.floor(omega.real.min() * 100) / 100, np.ceil(omega.real.max() * 100) / 100, dQ)
    psiLevels = np.arange(np.floor(omega.imag.min() * 100) / 100, np.ceil(omega.imag.max() * 100) / 100, dQ)
    
    ax1.contour(Z.real, Z.imag, omega.real, colors='b', linewidths=0.5, levels=phiLevels)
    ax1.contour(Z.real, Z.imag, omega.imag, colors='r', linewidths=0.5, levels=psiLevels)
    ax1.set_aspect(1)    
    ax1.set_ylabel('z')
    ax1.set_title(r"$\Omega(\zeta)=\frac{2Q}{\pi}\,\ln\left(2 \sin\left(i\, \frac{\pi\zeta}{2D}\right)\right)$")
    
    # --- omega1
    omega = omega - Q/D * Z
    
    # --- omega2
    omega = omega + 2 * Q / np.pi * np.log(2)
    
    phiLevels = np.arange(np.floor(omega.real.min() * 100) / 100, np.ceil(omega.real.max() * 100) / 100, dQ)
    psiLevels = np.arange(np.floor(omega.imag.min() * 100) / 100, np.ceil(omega.imag.max() * 100) / 100, dQ)    
    
    ax2.contour(Z.real, Z.imag, omega.real, colors='b', linewidths=0.5, levels=phiLevels)
    ax2.contour(Z.real, Z.imag, omega.imag, colors='r', linewidths=0.5, levels=psiLevels)
    ax2.set_aspect(1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('z')
    ax2.set_title(r"$\Omega(\zeta)=\frac{2Q}{\pi}\,\ln\left(2 \sin\left(i\, \frac{\pi\zeta}{2D}\right)\right)-\frac{Q}{D}\zeta$")
    
    fig.savefig(os.path.join(images, "radial_flow_deriv.png"))


# %% Complexe stroming in hoek
if __name__ == '__main__':
    
    b, D, N, k = 25, 10, 0.001, 0.025
    
    if True:
        comlexe_potentiaal(b=b, D=D, Q=1, d=0.1)    
    if False:
        stroming_analytisch(b=b, D=D, dxy=0.1, N=N, k=k, case=None, ax=None)
    if False:
        partial_penetraton(b=b, D=D, N=N, k=1, dxy=0.1)
    if False:
        partial_penenetration_anisotropie(b=2 * b, D=D, N=N, kh=25, kv=25/9, dxy=0.1)
    if False: # --- Numeric 1
        # Simulating the styled rectangular X-section after Ernst (1962, fig 7), in which the
        # flow domain is kept rectangular with constant thickness and the ditch is reduced
        # to a single (0.1x01 m) cell in the upper right corner. This setup corresponds
        # exactly to the vertical, horizontal and contraction of flow as theory predicts.
     
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.suptitle("Narekenen Ernst (1962, fig 7)\n"
                      f"b={b} m, D={D} m, k={k} m/d, N={N} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k):.5f} [d]")
        ax2.set_title("Stijghoogte aan freatisch vlak in de verschillende deelstroomfiguren")
        ax2.set(xlabel='x[m]', ylabel=r'$\phi - \phi_0$ [m]')
        ax2.grid(True)
           
        fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(12, 10))
        Q, dQ = N*b, N*b/20        
        fig.suptitle(f"Ernst (1962) Fig 7, numeriek. (Phi=blauw, Psi=Rood, dphi={dQ/k:.3f} m, dpsi={dQ} m2/d)\n"
                     f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k)} [d]")
        
        clrs = cycle('brgkmc')
        for ic, ax in enumerate(axs.flatten()):
            out = xsec_Ernst_rectangular_case(b=b, D=D, dxy=0.1, N=N, k=k, case=ic, ax=ax)            
            fig.savefig(os.path.join(images, "ErnstFig7_numeriek.png"))

            if ic in [3, 4, 5]: # ]> 0:
                clr = next(clrs)
                if ic + 1 == 3:
                    ax2.plot(out['gr'].xm[::5], out['Phi'][0, 0, ::5], '.-', color=clr, label=f"freatisch deelfiguur {ic + 1}")
                    ax2.plot(out['gr'].xm[::5], out['Phi'][-1, 0, ::5], '.--', color=clr, label=f"basis, deelfiguur {ic + 1}")
                else:
                    ax2.plot(out['gr'].xm, out['Phi'][0, 0, :], ls='solid', color=clr, label=f"freatisch deelfiguur {ic + 1}")
                    ax2.plot(out['gr'].xm, out['Phi'][-1, 0, :], ls='dashed', color=clr, label=f"basis, deelfiguur {ic + 1}")
        ax2.legend(loc='center')
        fig2.savefig(os.path.join(images, "ErnstNumeriekPhimaaiveld_456.png"))
        
    if False: # Numeric 2
        # Simulating the actual free-water table cross section of Ernst, including a real ditch
        # Using about the same sizes as Ernst (1962) did in his Fig. 7 in which he unraveled
        # the different flows (vertical, horizontal and caused by contraction of streamlines.)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        fig2.suptitle("Narekenen Ernst (1962, fig 7)\n"
                      f"b={b} m, D={D} m, k={k} m/d, N={N} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k)} [d]")
        ax2.set_title("Stijghoogte aan freatisch vlak in de verschillende deelstroomfiguren")
        ax2.set(xlabel='x[m]', ylabel=r'$\phi - \phi_0$ [m]')
        ax2.grid(True)
           
        fig, axs = plt.subplots(3,2, sharex=True, sharey=True, figsize=(12, 10))
        Q, dQ = N*b, N*b/20
        fig.suptitle(f"Ernst (1962) Fig 7, numeriek. (Phi=blauw, Psi=Rood, dphi={dQ/k:.3f} m, dpsi={dQ:.5f} m2/d)\n"
                     f"b={b} m, D={D} m, N={N} m/d, k={k} m/d, ND/(2k)={N * D/(2 * k):.3f} m, D/(2k)={D/(2*k)} [d]")
        
        clrs = cycle('brgkmc')
        for ic, ax in enumerate(axs.flatten()):        
            out = xsec_Ernst_watertable_case(b=b, D=D, dxy=0.1, N=N, k=k, case=ic, ax=ax)
            fig.savefig(os.path.join(images, "ErnstFig7_numeriek_freat.png"))

            if ic in [3, 4, 5]: # ]> 0:
                clr = next(clrs)
                ax2.plot(out['gr'].xm, out['Phi'][out['k0'], 0, out['cols']], ls='solid', color=clr, label=f"freatisch deelfiguur {ic + 1}")
                ax2.plot(out['gr'].xm, out['Phi'][-1, 0, :], ls='dashed', color=clr, label=f"basis, deelfiguur {ic + 1}")
        ax2.legend(loc='lower left')
        fig2.savefig(os.path.join(images, "ErnstNumeriekPhimaaiveld_456_freat.png"))

    plt.show()
    print("done")
