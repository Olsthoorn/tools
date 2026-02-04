# Edelman.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc, gamma
import pandas as pd

from fdm.src.fdm3t import fdm3t, dtypeQ, dtypeH
from fdm.src.mfgrid import Grid



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
        
    def s(self, A, n, x, t):
        """Return s and Q given that s0 = A * t ** (n/2)"""
        ie0 = ierfc(n, 0)
        u = x * np.sqrt(self.S / (4 * self.kD * t))
        s = A * t ** (n/2) * ierfc(n, u) / ie0
        Q = A / 2 * np.sqrt(self.kD * self.S) * t ** ((n-1)/2) * ierfc(n-1, u) / ie0
        return (s, Q)
    
    def Q(self, B, n, x, t):
        """Return Q and s given Q(0) = B t **(n/2)"""
        ie0 = ierfc(n+1, 0)
        u = x * np.sqrt(self.S / (4 * self.kD * t))
        Q = B * t ** (n/2) * ierfc(n, u) / ie0
        s = 2 * B / np.sqrt(self.kD * self.S) * t**((n+1)/2) * ierfc(n+1, u) / ie0
        return (Q, s)
    
    def u(self, t, x, eps=1e-20):
        assert np.all(np.isclose(np.diff(t), t[1]-t[0])), f'stepsize {t[1] - t[0]} not the same in t'
        return x * np.sqrt(self.S / (4 * self.kD * t.clip(eps, None)))
    
    def S_s0(self, t, x, s0=1):
        u_ = self.u(t, x)      
        return s0 * ierfc(0, u_) / ierfc(0, 0)
    
    def Q_s0(self, t, x, s0=1):
        u_ = self.u(t, x)
        return s0 / np.sqrt(t) * ierfc(-1, u_) / ierfc(0, 0) * np.sqrt(self.kD * self.S / 4)
    
    def S_q0(self, t, x, Q0=1):
        u_ = self.u(t, x)
        return Q0 * np.sqrt(t) * ierfc(1, u_) / ierfc(0, u_)  * 2 / np.sqrt(self.kD * self.S)
    
    def Q_q0(self, t, x, Q0=1):
        u_ = self.u(t, x)
        return Q0 * ierfc(0, u_) / ierfc(1, 0)
     
    def S_sat(self, t, x, a=1):
        u_ = self.u(t, x)        
        return a * t * ierfc(2, u_)  / ierfc(2, 0)
    
    def Q_sat(self, t, x, a=1):
        u_ = self.u(t, x)
        return a * np.sqrt(t) * ierfc(1, u_) / ierfc(2, 0) * np.sqrt(self.kD * self.S / 4)

    def S_qbt(self, t, x, b=1):
        u_ = self.u(t, x)
        return b * t * np.sqrt(t) * ierfc(3, u_) / ierfc(3, 0) * 2 / np.sqrt(self.kD * self.S)
    
    def Q_qbt(self, t, x, b=1):
        u_ = self.u(t, x)
        return b * t * ierfc(2, u_) / ierfc(3, 0)

    
    def S0(self, t, x, s0=1):        
        return (self.S_s0(t, x, s0=s0), self.Q_s0(t, x, s0=s0))
    
    def Q0(self, t, x, Q0=1):        
        return (self.S_q0(t, x, Q0=Q0), self.Q_q0(t, x, Q0=Q0))
    
    def Sat(self, t, x, a=1):        
        return (self.S_sat(t, x, a=a), self.Q_sat(t, x, a=a))

    def Qbt(self, t, x, b=1):        
        return (self.S_qbt(t, x, b=b), self.Q_qbt(t, x, b=b))

    def BR_S0(self, t, x, s0=1):
        SR, QR = self.S0(t, x, s0=s0)
        return SR[:-1] - SR[1:], QR[:-1], QR[1:]
    def BR_Q0(self, t, x, Q0=1):
        SR, QR = self.Q0(t, x, Q0=Q0)
        return SR[:-1] - SR[1:], QR[:-1], QR[1:]
    def BR_Sat(self, t, x, a=1):
        SR, QR = self.Sat(t, x, a=a)
        return SR[:-1] - SR[1:], QR[:-1], QR[1:]
    def BR_Qbt(self, t, x, b=1):
        SR, QR = self.Qbt(t, x, b=a)
        return SR[:-1] - SR[1:], QR[:-1], QR[1:]

def get_etable()->pd.DataFrame:    
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
    
    
def edelm(ie, u):
    """Return Edelman table values for case ie."""        
    if ie == 1:
        n = 0     
        return Fn(n, u)
    if ie == 2:
        n=0
        return Fn(n-1, u)
        # return np.sqrt(np.pi) / 2 / fn(n-1) * Fn(n-1, u)
    if ie == 3:
        n = 1
        return Fn(n, u)
        # return np.sqrt(np.pi) * fn(n-1) * Fn(n, u)
    if ie == 30:
        n=2
        return Fn(n-1, u)
        # return np.sqrt(np.pi) / 4 / fn(n-1) * Fn(n-1, u)
    if ie == 4:
        n = 3
        return Fn(n-1, u)
    if ie == 5:
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
    s0, Q0, a, b = 1, 1, 1, 1
    k, S , D = 1, 0.1, 10.
    kD, ss = k * D, S/D
    
    xp = 10.0
    dx = 10
    x = np.arange(-dx/2, 1000 + dx, dx)
    
    z=[0, -D]

    t = np.linspace(0, 200, 201)
    edel = Edelman(kD=kD, S=S)

    _, axs1 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    _, axs2 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    for i, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
        ax1.set(title=f"H change, kD={edel.kD}, S={edel.S}", xlabel='t [d]', ylabel='s [m]')        
        ax2.set(title=f"Q change, kD={edel.kD}, S={edel.S}", xlabel='t [d]', ylabel='Q [m2/d]')        
    
        if i == 0:
            SR, BR = edel.S0(t, xp, s0=s0)
            label=f"s0={s0}"
        if i == 1:
            SR, BR = edel.Q0(t, xp, Q0=Q0)
            label=f"Q0={Q0}"
        if i==2:
            SR, BR = edel.Sat(t, xp, a=a)
            label=f"a={a}"
        if i==3:
            SR, BR = edel.Qbt(t, xp, b=b)
            label=f"b={b}"
        ax1.plot(t, SR, label=label)
        ax2.plot(t, BR, label=label)
        
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()

        
    gr = Grid(x, [-0.5, 0.5], z)
    
    # --- index of xp
    ix = np.argmin(np.abs(gr.xm - xp))

    idomain = gr.const(1, dtype=int)
    hi = gr.const(0.)
    hi[:, 0, 0] = 1
    ss = gr.const(S/D)
    K = gr.const(k)
    
    for i, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
        if i == 0:
            fq = None
            FH = np.zeros(len(t), dtype=dtypeH)
            FH['h'] = s0
            FH['I'] = 0
            fh = {i:FH[i] for i in range(len(FH))}
            label=f's0={s0}'
        if i == 1:
            fh = None
            FQ = np.zeros(1, dtype=dtypeQ)
            FQ['q'] = Q0
            FQ['I'] = 0
            fq = {i:FQ[i] for i in range(len(FQ))}
            label=f'Q0={Q0}'
        if i == 2:
            FH = np.zeros(len(t), dtype=dtypeH)
            FH['h'] = a * t
            FH['I'] = 0
            fh = {i:FH[i] for i in range(len(FH))}
            fq = None
            label=f'a={a}'
        if i == 3:
            fh = None
            FQ = np.zeros(len(t), dtype=dtypeQ)
            FQ['q'] = b * t
            FQ['I'] = 0
            fq = {i:FQ[i] for i in range(len(FQ))}
            label=f'b={b}'

        out= fdm3t(gr, t=t, k=(K, K, K), ss=ss, fh=fh, fq=fq, hi=hi, idomain=idomain)
        ax1.plot(t, out['Phi'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix - 1], label=label)
        
        ax1.legend()
        ax2.legend()
        
def model_edelman2():
    s0, Q0, a, b = 1, 1, 1, 1
    k, S , D = 1, 0.1, 10.
    kD, ss = k * D, S/D
    
    xp = 250.0
    dx = 2
    x = np.hstack((-0.01, np.arange(0, 1000 + dx, dx)))
    
    z=[0, -D]

    gr = Grid(x, [-0.5, 0.5], z)
    
    # --- index of point xp
    ix = np.argmin(np.abs(gr.xm - xp))

    idomain = gr.const(1, dtype=int)
    hi = gr.const(0.)
    hi[:, 0, 0] = 1
    ss = gr.const(S/D)
    K = gr.const(k)
 
    
    t = np.linspace(0, 200, 201)
    edel = Edelman(kD=kD, S=S)

    _, axs1 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    _, axs2 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    for n, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
        ax1.set(title=f"H change, kD={edel.kD}, S={edel.S} , xp={xp}, ix={ix}", xlabel='t [d]', ylabel='s [m]')
        ax2.set(title=f"Q change, kD={edel.kD}, S={edel.S} , xp={xp}, ix={ix}", xlabel='t [d]', ylabel='Q [m2/d]')
    
        if n == 0:
            A = s0            
            label=f"n={n}, "r"$s_0=A t^n/2, Q0=A t^{\left(n-1\right)/2}\left(\frac{1}{f\left(n-1\right)}\sqrt{\frac{kDS}{4}}\right)$"
        elif n == 1:
            B = Q0
            label=f"n={n}: "r"Q_0=Q_0 " + r"$s_0=Q_0 t^{n/2}\left(f\left(n-1\right)\sqrt{\frac{4}{kDS}}\right)$"
        elif n==2:
            A=a            
            label=r"n={n}: " + r"$s_0=a t^{\left(n-1\right)/2}\left(\frac{1}{f\left(n-1\right)}\sqrt{\frac{kDS}{4}}\right)$"
        elif n==3:
            B = b
            label=f"n={n}: " + r"$Q_0 = b t^{{n-1}{2}}$ " + r"$Bt^{n/2}\left(f\left(n-1\right)\sqrt{\frac{4}{kDS}}\right)$"

        u_ = edel.u(t=t, x=xp)

        if n in [0, 2]:
            ax1.plot(t, A * t ** (n/2) * Fn(n, u_), label=label)
            ax2.plot(t, A * t ** ((n-1)/2) * np.sqrt(kD * S/ 4) * Fn(n-1, u_) / fn(n=1), label=label)
        else:
            ax1.plot(t, B * t**(n/2) *  np.sqrt(4 / (kD * S)) * fn(n-1) * Fn(n, u_),label=label)
            ax2.plot(t, B * t**((n-1)/2) * Fn(n-1, u_), label=label)
    
        if n == 0:
            fq = None
            FH = np.zeros(len(t), dtype=dtypeH)
            FH['h'] = s0 * t**(n/2)
            FH['I'] = 0
            fh = {i:FH[i] for i in range(len(FH))}
            label=f's0={s0}'
        elif n == 1:
            fh = None
            FQ = np.zeros(len(t), dtype=dtypeQ)
            FQ['q'] = Q0 * t ** (n-1)/2
            FQ['I'] = 0
            fq = {i:FQ[i] for i in range(len(FQ))}
            label=f'Q0={Q0}'
        elif n == 2:
            FH = np.zeros(len(t), dtype=dtypeH)
            FH['h'] = a * t**(n/2)
            FH['I'] = 0
            fh = {i:FH[i] for i in range(len(FH))}
            fq = None
            label=f'a={a}'
        elif n == 3:
            fh = None
            FQ = np.zeros(len(t), dtype=dtypeQ)
            FQ['q'] = b * t**(n-1)/2
            FQ['I'] = 0
            fq = {i:FQ[i] for i in range(len(FQ))}
            label=f'b={b}'

        out= fdm3t(gr, t=t, k=(K, K, K), ss=ss, fh=fh, fq=fq, hi=hi, idomain=idomain)
        ax1.plot(t, out['Phi'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix - 1], label=label)
        
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()

        pass
 
         
def model_edelman3():
    s0, Q0, a, b = 1, 1, 1, 1
    k, S , D = 1, 0.1, 10.
    kD, ss = k * D, S/D
    A = 1.
    
    xp = 250.0
    dx = 2
    x = np.hstack((-0.01, np.arange(0, 1000 + dx, dx)))
    
    z=[0, -D]

    gr = Grid(x, [-0.5, 0.5], z)
    
    # --- index of point xp
    ix = np.argmin(np.abs(gr.xm - xp))

    idomain = gr.const(1, dtype=int)
    hi = gr.const(0.)
    hi[:, 0, 0] = 1
    ss = gr.const(S/D)
    K = gr.const(k)
 
    
    t = np.linspace(0, 200, 201)
    edel = Edelman(kD=kD, S=S)

    _, axs1 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    _, axs2 = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    for n, (ax1, ax2) in enumerate(zip(axs1.ravel(), axs2.ravel())):
        ax1.set(title=f"H change, kD={edel.kD}, S={edel.S} , xp={xp}, ix={ix}", xlabel='t [d]', ylabel='s [m]')
        ax2.set(title=f"Q change, kD={edel.kD}, S={edel.S} , xp={xp}, ix={ix}", xlabel='t [d]', ylabel='Q [m2/d]')
    
        u_ = edel.u(t=t, x=xp)

        ax1.plot(t, A * t ** (n/2) * Fn(n, u_), label=r"$A t^{{\frac{n}{2}}}$")
        ax2.plot(t, A * t ** ((n-1)/2) * np.sqrt(kD * S/ 4) * Fn(n-1, u_) / fn(n=1), label=r"$A t^{{\frac{n}{2}}}$")
    
        fq = None
        FH = np.zeros(len(t), dtype=dtypeH)
        FH['h'] = A * t**(n/2)
        FH['I'] = 0
        fh = {i:FH[i] for i in range(len(FH))}
        label=f'A={A}'

        out= fdm3t(gr, t=t, k=(K, K, K), ss=ss, fh=fh, fq=fq, hi=hi, idomain=idomain)
        ax1.plot(t, out['Phi'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix], label=label)
        ax2.plot(t[1:], out['Qx'][:, 0, 0, ix - 1], label=label)
        
        ax1.grid()
        ax2.grid()
        ax1.legend()
        ax2.legend()

        pass


if __name__ == '__main__':
    if False:
        etable = get_etable()
        btable = compare(etable=etable)
        print(etable)
        print(btable)
        show_ierfc()
        pass
    if False:
        model_edelman2()
    if True:
        model_edelman3()
    if False:
        for n in range(5):
            print(f"n={n}: ierfc(n,0)={ierfc(n,0)}, 2**n * gamma(n/2 + 1)={1/(2**n * gamma(n/2+1))}")
    plt.show()

    
    
    