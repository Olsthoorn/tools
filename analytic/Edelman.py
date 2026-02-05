# Edelman.py
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, gamma
import pandas as pd

from fdm.src.fdm3t import fdm3t, dtypeQ, dtypeH
from fdm.src.mfgrid import Grid


# --- Directory to store images
images = os.path.join(*(Path(os.getcwd()).parts[:Path(os.getcwd()).parts.index('tools') + 1] + ('analytic', 'images')))


def ierfc(n:int, u:float | np.ndarray)->float|np.ndarray:
    """Return repeated intefral of the complementary error function."""
    assert n >= -1, f'n must be >= -1, not {n}'
    if n == -1:
        return 2 / np.sqrt(np.pi) * np.exp(-u ** 2)
    if n == 0:
        return erfc(u)
    
    return -u/n * ierfc(n-1, u) + 1 / (2*n) * ierfc(n-2, u)

def Fn(n, u):
    return ierfc(n, u) / ierfc(n, 0)

def fn(n):
    return ierfc(n+1, 0) / ierfc(n, 0)

class Edelman:
    def __init__(self, kD, S):
        self.kD = kD
        self.S = S
    
    def u(self, t, x, eps=1e-20):
        assert np.all(np.isclose(np.diff(t), t[1]-t[0])), f'stepsize {t[1] - t[0]} not the same in t'
        return x * np.sqrt(self.S / (4 * self.kD * t.clip(eps, None)))
    
    # --- further methods in this class deleted because obsolete
    

def get_etable()->pd.DataFrame:
    """Return DataFrame with some Edelman values from Huisman (1972, p42-47)""" 
    idx = [0, 0.1, 0.3, 0.65, 1.0, 1.6]
    data = [(1.0000, 1.0000, 1.0000, 1.0000, 1.0000),
        (0.8875, 0.9900, 0.8327, 0.7935, 0.7624),
        (0.6714, 0.9139, 0.5569, 0.4829, 0.4286),
        (0.3580, 0.6554, 0.2430, 0.1798, 0.1394),
        (0.1573, 0.3679, 0.0891, 0.0568, 0.0388),
        (0.0237, 0.0773, 0.0102, 0.0052, 0.0029), 
    ]
    columns = ['E1', 'E2', 'E3', 'E4', 'E5']
    return pd.DataFrame(index=idx, data=data, columns=columns)

def compare(etable, cols=[1, 2, 3, 30, 4, 5]):
    """
    Return table with Edelman values to compare with the values
    in Huisman (1972, p42-47).
    
    Now that we're proven correct, we don't need this anymore.
    Notice that in Huisman's book E2 and E1 are switched for
    some reason as they correcpond to F(-1) and F(0) resp.
    in stead of F(0) and F(-1).
    
    E1 = ierfc( 0, u) / ierfc(0, 0)
    E2 = ierfc(-1, u) / ierfc(0, 0) * np.sqrt(np.pi) * (1/2)
    E3 = ierfc( 1, u) / ierfc(2, 0) * np.sqrt(np/pi)  * (1/4)          
    E4 = ierfc( 2, u) / iefrc(2, 0)        
    E5 = ierfc( 3, u) / ierfc(3, 0) * np.sqrt(np.pi) * (3/ 2)
    """
    u = np.asarray(etable.index)
    btable = pd.DataFrame(index=etable.index)
    
    for col in cols:
        btable[f'E{col}'] = edelm(col, u)
        
    return btable

def generate_edelman_tables():
    """Return a pandas table with the Edelman data.
    
    Note that the n are ordered, but E1 and E2 are
    switched in Huisman (1972)
    """
    idx = np.arange(0, 0.501, 0.1)
    tables = pd.DataFrame(index=idx)
    for n, E in zip(range(-1, 4), ['E2', 'E1', 'E3', 'E4', 'E4']):
        tables[f'n={n}: {E}'] = ierfc(n, idx) / ierfc(n, 0)
    return tables
    
    
def edelm(ie, u):
    """Return Edelman table values for case ie.
    
    Each Edelman function has it's own n. Use this
    n to computed the F(n, u) and F(n-1, u)
    
    This is to link/verify the relation between Ei and Fn
    """        
    if ie == 1:
        # --- E1
        n = 0     
        return Fn(n, u)
    if ie == 2:
        # --- E2
        n=0
        return Fn(n-1, u)
        # return np.sqrt(np.pi) / 2 / fn(n-1) * Fn(n-1, u)
    if ie == 3:
        # --- E3
        n = 1
        return Fn(n, u)
        # return np.sqrt(np.pi) * fn(n-1) * Fn(n, u)
    if ie == 30:
        # --- E3  Note that F(n-1,u) with n=2 is the same as F(n,u) with n=1
        n=2
        return Fn(n-1, u)
        # return np.sqrt(np.pi) / 4 / fn(n-1) * Fn(n-1, u)
    if ie == 4:
        # --- E4
        n = 3
        return Fn(n-1, u)
    if ie == 5:
        # --- E5
        n = 3
        return Fn(n,u)
        # return 1.5 * np.sqrt(np.pi) * fn(n-1) * Fn(n,u)
    
    raise ValueError(f"n={ie} must be one of [1, 2, 3, 30, 4, 40, 5]")

# --- show ierfc(n,u)
def show_ierfc():
    u = np.linspace(0, 3, 200)
    
    fig, ax = plt.subplots()
    ax.set(title='erfc(n, u)', xlabel='u', ylabel='ierfc(n,u)')
    
    for n in range(-1, 4):
        ax.plot(u, ierfc(n, u), label=f'erfc({n}, u)')
    ax.grid()
    ax.legend()
    return ax
 
         
def model_edelman():
    """Compute Edelman curves anlytically and by fdm  model."""
    
    # --- Data
    k, S , D = 1, 0.1, 10.
    kD, ss = k * D, S/D
    A = 1.
        
    # --- fdm grid
    dx = 2
    x = np.hstack((-0.01, np.arange(0, 1000 + dx, dx)))    
    z=[0, -D]
    gr = Grid(x, [-0.5, 0.5], z)
    
    # --- Obs point
    xp = 50.0

    # --- index of point xp
    ix = np.argmin(np.abs(gr.xm - xp))

    # --- Model arrays
    idomain = gr.const(1, dtype=int)
    hi = gr.const(0.)
    hi[:, 0, 0] = 1
    ss = gr.const(S/D)
    K = gr.const(k)
 
    # --- simulation time
    t = np.linspace(0, 200, 201)
    
    edel = Edelman(kD=kD, S=S)

    # --- Setup two figure with for axes each for n in [0, 1, 2, 3]
    fig1, axs1 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    fig2, axs2 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    
    fig1.suptitle(
        r"Head $\phi$ [m]" + "\n"
        fr"$A$={A}, $L$={gr.x[-1]:.0f}, $dx$={gr.dx[-1]}, $kD$={kD}:.0f, $S$={S}, $x_p$={xp:.0f} m $ix$={ix}"
    )
    fig2.suptitle(
        r"Flow $Q$ [m2/d]" + "\n"
        fr"$A$={A}, $L$={gr.x[-1]:.0f}, $dx$={gr.dx[-1]}, $kD$={kD}:.0f, $S$={S}, $x_p$={xp:.0f} m $ix$={ix}"
    )

    # --- Loop over n in 0..3
    for n, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
        
        # --- Axes labels and title
        ax1.set(title=fr"$\phi(0,t)=A t^{{{n/2}}}$", xlabel=r'$t$ [d]', ylabel=r'$\phi(x_p,t)$ [m]')
        
        qtitle = fr"$Q(0,1) = \sqrt{{\frac{{kDS}}{{4}}}}\frac{{F(({n-1}),0)}}{{f({n-1})}}$"
        ax2.set(title=qtitle, xlabel=r'$t$ [d]', ylabel=r'$Q(x_p,t$ [m2/d]')
   
        # --- Compute curves analytically 
        u_ = edel.u(t=t, x=xp)
        
        ax1.plot(t, A * t ** (n/2) * Fn(n, 0), label=fr"$\phi(0,t) = A t^{{ {n/2} }}$")
        ax1.plot(t, A * t ** (n/2) * Fn(n, u_), label=fr"$\phi({xp:.0f},t) = A t^{{ {n/2} }}F({n},u)$")
        ax2.plot(t, A * t ** ((n-1)/2) * np.sqrt(kD * S/ 4) * Fn(n-1, 0) / fn(n-1),
                 label=fr"$Q(0,t) = A t^{{ {(n-1)/2} }} \sqrt{{\frac{{kDS}}{{4}}}}\frac{{F({n-1},0)}}{{f({n-1})}}$")
        ax2.plot(t, A * t ** ((n-1)/2) * np.sqrt(kD * S/ 4) * Fn(n-1, u_) / fn(n-1),
                 label=fr"$Q({xp:.0f},t) = A t^{{ {(n-1)/2} }} \sqrt{{\frac{{kDS}}{{4}}}}\frac{{F({n-1},u)}}{{f({n-1})}}$")
        
        # --- Boundary arrays for each model
        fq = None
        FH = np.zeros(len(t), dtype=dtypeH)
        FH['h'] = A * t**(n/2)
        FH['I'] = 0
        fh = {i:FH[i] for i in range(len(FH))}

        # --- Simulate the fdm model
        out= fdm3t(gr, t=t, k=(K, K, K), ss=ss, fh=fh, fq=fq, hi=hi, idomain=idomain)
        
        # --- Plot model results for xp
        ax1.plot(t[::10], out['Phi'][::10, 0, 0, ix], 'o', label=fr'$\phi({xp:.0f},t)$, model')
        ax2.plot(t[1::10], out['Qx'][::10, 0, 0, ix], 'o', label=fr'$Q({xp:.0f},t)$, model')
    
        # --- Put text so that it's clear what n is in each plot
        xt, yt = [0.1, 0.1, 0.1, 0.1], [0.65, 0.65, 0.65, 0.65]
        bbox = dict(boxstyle="round", facecolor='white', edgecolor='black')
        ax1.text(xt[n], yt[n], f"n={n}", bbox=bbox, transform=ax1.transAxes)
        ax2.text(xt[n], yt[n], f"n={n}", bbox=bbox, transform=ax2.transAxes)
    
        # --- Round-off
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()
        
        fig1.savefig(os.path.join(images, "Edelman_heads.png"))
        fig2.savefig(os.path.join(images, "Edelman_flows.png"))
        

if __name__ == '__main__':
    if False:
        etable = get_etable()
        btable = compare(etable=etable)
        print(etable)
        print(btable)
        show_ierfc()
        pass
    if False:
        print("Eleman tables:")
        tables = generate_edelman_tables()
        print(tables)
    if True:
        model_edelman()
    if False:
        for n in range(5):
            print(f"n={n}: ierfc(n,0)={ierfc(n,0)}, 2**n * gamma(n/2 + 1)={1/(2**n * gamma(n/2+1))}")
    plt.show()

    
    
    