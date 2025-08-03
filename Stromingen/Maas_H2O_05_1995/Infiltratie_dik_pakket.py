# %% Dik freatisch pakket infiltreert niet lekker (H2O Nr. 5 1995) door Kees Maas

# The problem:
# Een infiltratiesloot die afvoert heeft een debiet dat lager is naarmate het pakket dikker is.


import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytical solution
# #############################
kh, kv = 10., 10.
h, b = 10., 20.
d = 1. # Step size grid
Q = 1.0 # m2/d

x = np.linspace(-10., 10., 101)
y = np.linspace(-10.,  0.,  51)

X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
h = 10.
Phi = np.log(X + 1j * Y)
t = np.exp(np.pi * Z / h)

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.plot_surface(X, Y, Phi.real, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)
ax.plot_surface(X, Y, Phi.imag, edgecolor='red', lw=0.5, rstride=8, cstride=8,
                alpha=0.3)

fig, ax = plt.subplots(figsize=(10, 6))

ax.set(aspect=1.)
ax.contour(Z.real, Z.imag, Phi.real, colors='royalblue', levels=20)
ax.contour(Z.real, Z.imag, Phi.imag, colors='red', levels=20)

fig, ax = plt.subplots(figsize=(10, 6))
 
ax.contour(Z.real, Z.imag, t.real, colors='royalblue', levels=100)
ax.contour(Z.real, Z.imag, t.imag, colors='red', levels=100)

ax.set(aspect=1.)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytische transformatie voor w = z ** n
############################################

fig, axs = plt.subplots(2, 3, sharex=True, sharey=True,figsize=(10, 6))
fig.suptitle(r'Analytical transformation for $z = w^n$')
axs[0, 0].set_ylabel('Im(z)')
axs[1, 0].set_ylabel('Im(z)')
axs[1, 0].set_xlabel('Re(z)')
axs[1, 1].set_xlabel('Re(z)')
axs[1, 2].set_xlabel('Re(z)')

# Kies de Phi en Psi lijnen in het Omega vlak
Q, k = 1.0, 10.
N = 20
phi = np.linspace(-Q, Q, 2 * N + 1) # We can express the extents just as well in terms of Q.
psi = np.linspace(0, Q, N + 1)
Phi, Psi = np.meshgrid(phi, psi)
Omega = Phi + 1j * Psi

levels = np.linspace(-Q, Q, 2 * N + 1) # Q will be zero at Phi = 0 so start with -Q, to Q

# Teken dan de isolijnen voor Phi en Psi in het z-vlak
for ax, n in zip(axs.flat, [3., 2., 1, 1/2, 1/3, 1/4]):
    z = Omega ** (n)
    ax.set_title(r"$z=\Omega^{n:.3g}$")
    ax.contour(z.real, z.imag, Omega.real,linewidths=0.25, colors='royalblue', levels=levels)
    ax.contour(z.real, z.imag, Omega.imag,linewidths=0.25, colors='red', levels=levels)
    ax.set_title(f'n = {n:.2g}')
    ax.set(aspect=1.)
    ax.grid()
    
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Conformal mapping for Omega = Q ln(z)
#######################################
# We'll take a grid of z values and contour the isolines of Omega in the z-plane.

x = np.linspace(-1, 1, 51)
y = np.linspace(-1, 1, 51)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
Omega= Q * np.log(Z)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Conformal mapping for $w = \log(z)$')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

csphi = ax.contour(Z.real, Z.imag, Omega.real, lw=0.5, colors='royalblue', levels=20)
cspsi = ax.contour(Z.real, Z.imag, Omega.imag, lw=0.5, colors='red', levels=20)

csphi.clabel(fmt='%1.2f', fontsize=8, colors='royalblue')
cspsi.clabel(fmt='%1.2f', fontsize=8, colors='red')

ax.set(aspect=1.)

plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Conformal mapping for Omega = Q ln(z) or z = exp(Omega / Q)
##################################
# We'll take a grid of Omega values and contour the isolines of z in the z-plane.

Q = 1.0
N = 100
phi = np.linspace(0, 2 * np.pi * Q, N + 1)
psi = np.linspace(0, 2 * np.pi * Q, N + 1)
Phi, Psi = np.meshgrid(phi, psi)
Omega = Phi + 1j * Psi
z = np.exp(Omega / Q)
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'Conformal mapping for $z = \exp(\Omega / Q)$')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
levels = np.linspace(0, 2 * np.pi * Q, 37)

csphi = ax.contour(z.real, z.imag, Omega.real, lw=0.5, colors='royalblue', levels=levels)
cspsi = ax.contour(z.real, z.imag, Omega.imag, lw=0.5, colors='red', levels=levels)
csphi.clabel(fmt='%1.2f', fontsize=8, colors='royalblue')
cspsi.clabel(fmt='%1.2f', fontsize=8, colors='red')

ax.set_aspect(1.)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Conformal mapping for w = sin(z)
##################################

# We'll take a grid of z values and contour the isolines of Omega in the z-plane.

x = np.linspace(-np.pi / 2, np.pi / 2, 51)
y = np.linspace(0, np.pi, 51)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

Omega= np.sin(Z)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Conformal mapping for $\Omega = \sin(z)$' + "\nContouring Phi and Psi on the z-plane")
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set(aspect=1.)

csphi = ax.contour(Z.real, Z.imag, Omega.real, lw=0.5, colors='royalblue', levels=20)

cspsi = ax.contour(Z.real, Z.imag, Omega.imag, lw=0.5, colors='red', levels=20)

csphi.clabel(fmt='%1.2f', fontsize=8, colors='royalblue')
cspsi.clabel(fmt='%1.2f', fontsize=8, colors='red')

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Conformal mapping for $\Omega = \sin(z)$' + '\nPlotting lines of Phi and Psi')
ax.set_xlabel('Re(Omega)')
ax.set_ylabel('Im(Omega)')

ax.plot(Omega.real, Omega.imag, color='blue')
ax.plot(Omega.T.real, Omega.T.imag, color='red')
ax.legend()

ax.set(aspect=1.)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Conformal mapping for z = sin(w)
##################################

# We'll contour Omega in the z-plane and we'll directly plot Omega lines in the same z-plane with perfect overlap.

# Generate an Omega grid
phi = np.linspace(-np.pi / 2, np.pi / 2, 51)
psi = np.linspace(0, np.pi / 2, 51)
Phi, Psi = np.meshgrid(phi, psi)
Omega = Phi + 1j * Psi

Z = np.sin(Omega)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title('Conformal mapping for $w = \sin(z)$, two ways, with perfcect overlap')
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')

# Contouring Omega in the z-plane
ax.contour(Z.real, Z.imag, Omega.real, linewidths=3, colors='royalblue', levels=phi)
ax.contour(Z.real, Z.imag, Omega.imag, linewidths=3, colors='red', levels=psi)

# Plotting  the Omega lines in the z-plane
ax.plot(Z.real, Z.imag, 'y', lw=1, )
ax.plot(Z.T.real, Z.T.imag, 'w', lw=1)

ax.set(aspect=1.)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytische transformatie voor flow in a infinite layer (Verruijt 1970. p111-112)
#####################################

def inflayer(Omega, h, Q):
    """Analytical transformation for flow in a infinite layer."""
    return h / np.pi * np.log((np.cosh(np.pi * Omega / Q) - 1) / 2)

h, Q = 10., 1.
phi = np.linspace(0, 1, 51)
psi = np.linspace(0, Q, 51)
Phi, Psi = np.meshgrid(phi, psi)
Omega = Phi + 1j * Psi
z = inflayer(Omega, h, Q)
fig, ax = plt.subplots(figsize=(10, 6))
ax.contour(Omega.real, Omega.imag, z.real, colors='royalblue', levels=20)
ax.contour(Omega.real, Omega.imag, z.imag, colors='red', levels=20)

ax.set(aspect=1.)
plt.tight_layout()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytische transformatie voor flow in a contraction (Verruijt 1970. p117-118)
##########################################

# Just contour the Phi and Psi on the z-grid

def contraction(z, p, a, Q):
    """Analytical transformation for flow in a contraction."""
    return - 2 * 1j * Q / np.pi * np.arcsin(np.sin(np.pi * z / (2 * a)) / np.sin(np.pi * p / (2 * a)))

a, p, Q = 10., 2., 1.
x = np.linspace(-a, a, 101)
y = np.linspace(0, 5*p, 251)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y
Phi = contraction(Z, p, a, Q)
fig, ax = plt.subplots(figsize=(10, 6))

ax.contour(Z.real, Z.imag, Phi.real, colors='royalblue', levels=20)
ax.contour(Z.real, Z.imag, Phi.imag, colors='red', levels=20)

ax.set_aspect(1.)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytical solution for flow into a ditich of width b into an aquifer of half-infinite extent

# Method 1: Drawing the Phi and Psi lines in the z-plane, starting with a grid of Omega coordinates
# =====================================

def get_points(H, b=None, a=0.):
    """Get the points A, B, Ctop, Cbot, D for the conformal mapping."""
    Ctop = 1.5 * b
    B =    b
    A =    a
    D =    0.      -1j * H   
    Cbot = 1.5 * b - 1j * H
    return ('A', 'B', 'Ctot', 'Cbot', 'D'), (A, B, Ctop, Cbot, D)

def zeta_plane(z=None, H=None):
    return np.sin(np.pi / H * (1j * z - H / 2))
    
def get_pq(H, b=None, a=0.):
    """Calculate p and q for the conformal mapping."""
    ztaPnts = zeta_plane(np.array(get_points(H, b=b, a=a)[1]), H)
    ztaA, ztaB = ztaPnts[[0, 1]]    
    p = 2 / (ztaA - ztaB)
    q = -(ztaA + ztaB) / (ztaA - ztaB)
    return p, q

def z_plane(Omega, Q=None, b=None, a=0., H=None):    
    p, q = get_pq(H, b=b, a=a)
    sin = np.sin(1j * np.pi / Q * Omega + np.pi / 2)    
    return -1j * H * (1 / 2 + 1 / np.pi * np.arcsin((sin-q) / p))
    
Q, H, b, a = 1.0, 10., 25., 20.
nms, pnts = get_points(H, b)

# Because we draw the lines (no contouring) use many points
# and only plot every 10th line.
phi = np.linspace(0, 2 * Q, 501)
psi = np.linspace(0, 1 * Q, 251)
phi[0] += 1e-6
psi[0] += 1e-6
psi[-1] = Q -1e-6
Phi, Psi = np.meshgrid(phi, psi)
Omega = Phi + 1j * Psi

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'Potential and Streamlines')
ax.set_xlabel(r'$\Re(z)$')
ax.set_ylabel(r'$\Im(z)$')

Z = z_plane(Omega, Q=Q, b=b, a=a, H=H)

# Plot the marker points
for nm, pnt in zip(nms, pnts):
    ax.plot(pnt.real, pnt.imag, 'o', label=nm)
ax.legend()

lines1 = ax.plot(Z.real[:, ::10], Z[:, ::10].imag, color='blue')
lines2 = ax.plot(Z.T[1:, ::10].real, Z.T[1:, ::10].imag, 'red')

ax.set_aspect(1)
plt.tight_layout()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytical solution for flow into a ditich of width b into an aquifer of half-infinite extent

# Method 2: Contouring Phi and Psi in the z-plane (using a grid of z-coordinates)
# =====================================

def get_points(H, b=None, a=0.):
    """Get the points A, B, Ctop, Cbot, D for the conformal mapping."""
    Ctop = 1.5 * b
    B =    b
    A =    a
    D =    0.      -1j * H   
    Cbot = 1.5 * b - 1j * H
    return ('A', 'B', 'Ctot', 'Cbot', 'D'), (A, B, Ctop, Cbot, D)

def w_plane(z, Q=None, b=None, a=0., H=None):
    p, q = get_pq(H, b=b, a=a)
    sin = np.sin(np.pi / H * (1j * z - H / 2))    
    return -1j * Q / np.pi * (np.arcsin(p * sin + q) - np.pi/ 2)
    
def get_Z(x, y):
    """Return grid in the z-plane"""
    y[ 0] += 0.001 # keep bottom streamline inside model
    y[-1] -= 0.001 # keep top streamline inside model   
    X, Y = np.meshgrid(x, y)    
    return X + 1j * Y


Q, H, b, a = 1.0, 10., 25., 20.
nms, pnts = get_points(H, b)

x = np.linspace(0, 4 * H, 51)
y = np.linspace(-H, 0, 26)
Z = get_Z(x, y)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(r'Potential and Streamlines')
ax.set_xlabel(r'$\Re(z)$')
ax.set_ylabel(r'$\Im(z)$')

Omega = w_plane(Z, Q=Q, b=b, a=a, H=H)

# Plot the marker points
for nm, pnt in zip(nms, pnts):
    ax.plot(pnt.real, pnt.imag, 'o', label=nm)
ax.legend()

phi_levels = np.linspace(0, 2 * Q, 21) 
psi_levels = np.linspace(0, Q, 21)

cs1 = ax.contour(Z.real, Z.imag, Omega.real, colors='blue', levels=phi_levels)
cs2 = ax.contour(Z.real, Z.imag, Omega.imag, colors='red', levels=psi_levels)

ax.set_aspect(1.0)
plt.tight_layout()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Analytical solution for flow into a ditich of width b into an aquifer of half-infinite extent

# Method 2: Contouring Phi and Psi in the z-plane (using a grid of z-coordinates)

# Now we'll compute the head in the ditch relative to  a point 5 H to
# the left of the ditch to see how much the ris in head of the ditch is
# as a function of H given it infitlrates a fixed flow Q.
# =====================================

def get_zPnts():
    """Get the points A, B, Ctop, Cbot, D for the conformal mapping.""" 
    global H, b, a, case
    if case == 'sin':        
        zPnts = {   
            'Rt': a + b + 20,              # right top                        
            'Dr': a + b,              # Ditch right            
            'Dl': a,                  # Fixing the ditch to let only H grow 
            'Lt': 0.,                 # Left top
            'Lm': 0. - 1j * 0.5 * H,     # Left middle
            'Lb': 0. - 1j * H,           # Left bottom            
            'Rb': a + b + 20 - 1j * H,     # right bottom
        }
    elif case == 'exp':
        zPnts ={
            'Rt': a + b + 20,         # right top
            'Dr': a + b,              # Ditch right
            'Dl': a,                  # Fixing the ditch to let only H grow 
            'Lt': -np.inf,      # Left top   
            'Lb': -np.inf - 1j * H,       # Left bottom
            'Rb': a + b + 20  - 1j * H,       # right bottom
        }
    else:
        raise ValueError(f"Unknown case {case}")
    return zPnts

def get_pq():
    """Calculate p and q for the conformal mapping."""
    global H, a, b
    zPnts = get_zPnts()
    ztaPnts = {k: get_zeta(z) for k, z in zPnts.items()}
    ztaDr, ztaDl = ztaPnts['Dr'], ztaPnts['Dl']
    p = 2 / (ztaDl - ztaDr)
    q = 1 - p * ztaDl
    return p, q

def get_Z(tol=1e-6):
    """Return grid in the z-plane"""
    global H, a, b, case
    

    if case == 'sin':
        x = np.linspace(0, 60, 241)
        x[0] += 1e-6 # Keep x completely inside section
    elif case == 'exp':
        x = np.linspace(0, 60, 241)
    else:
        raise ValueError(f"Unknown case {case}")
    
    y = np.linspace(-H, 0, 4 * int(H) + 1)
    # Keep streamlines completely inside the section
    y[ 0] += tol
    y[-1] -= tol
    
    X, Y = np.meshgrid(x, y)    
    return X + 1j * Y

def get_zeta(z):
    """Scale z and clap-out the lines +/- pi/2"""
    global H, case
    if case == 'sin':
        return np.sin(1j * np.pi / H * (z + 1j * H / 2))
    elif case == 'exp':
        return np.exp(np.pi  / H * (z + 1j * H))
    else:
        raise ValueError(f"Unknown case {case}")
    
def get_zeta1(z):    
    p, q = get_pq()
    return p * get_zeta(z) + q

def get_zeta2(z):
    return np.arcsin(get_zeta1(z))

def get_w(z):
    global Q    
    return -1j * Q / np.pi * (get_zeta2(z) - np.pi / 2)


def plot_marker_points(Pnts=None, ax=None):
    """Plot marker points on desired plane.
    
    Pnts: dictionary of nm, pnt
    Pnts should be in the coordinate of the plane.
    ax: plt.Axes on which to plot              
    """
    global H
    for nm, pnt in Pnts.items():
        ax.plot(pnt.real, pnt.imag, 'o', label=nm)
    return ax

def w_inf():
    global b, H
    g = np.exp(np.pi * b  / H)
    q = -(1 + g) / (1 - g)
    return -1j * Q / np.pi * (np.arcsin(q + 0j)-np.pi / 2)
  
# %%
case = 'exp'

Hs = [10, 20, 30]

fig, axs = plt.subplots(len(Hs), 1, sharex=True, figsize=(10, 16))

fig1, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_title(f"Phi along top and bottom for different aquifer H (case={case})")
ax1.set_xlabel('Re(z) [m]')
ax1.set_ylabel('Phi [m2/d]')
ax1.grid(True)
clrs = cycle('brgkmc')

phiLm = dict()
phinf = dict()

for H, ax in zip(Hs, axs):
    
    # Fix Q and the left corner and width of the ditch
    Q = 1.0 # m2/d
    a = 30. # x of left side of ditch
    b = 10. # width of ditch

    zPnts = get_zPnts()
        
    Z = get_Z()
        
    ax.set_title(f'Potential and Streamlines. case={case}.  H={H} m, xDleft={a} m, xDright={a + b} m')
    ax.set_xlabel(r'$x=\Re(z)$')
    ax.set_ylabel(r'$y=\Im(z)$')
    
    x = Z.real[0]
    y = Z.imag.T[0]
    ax.set_xlim(x[0] -1, x[-1] + 1)
    ax.set_ylim(y[0] -1, y[-1] + 1)

    Omega = get_w(Z)

    # Contouring levels
    phi_levels = np.linspace(0, 3 * Q, 31) 
    psi_levels = np.linspace(0, Q, 11)
    phi_levels[0] = 0.001 * Q
    psi_levels[ 0] = 0.00001 * Q
    psi_levels[-1] = 0.99990 * Q

    # Contour potential and stream function
    cs1 = ax.contour(Z.real, Z.imag, Omega.real, colors='blue', levels=phi_levels)
    cs2 = ax.contour(Z.real, Z.imag, Omega.imag, colors='red', levels=psi_levels)

    cs1.clabel(fmt='%1.2f')
    cs2.clabel(fmt='%1.2f')

    plot_marker_points(zPnts, ax)

        
    # Marker point where to compute the potential
    pm = 0.0001 - 0.5j * H    
    plot_marker_points({'Ref': pm}, ax)

    phiLm[H] = np.round(get_w(pm).real, 3)
    phinf[H] = np.round(w_inf(), 3)
    
    ax.text(pm.real + 0.1 * H, pm.imag, fr"$\Phi = ${phiLm[H]:.3f}$\,m^2/d$")
    
    ax.set_aspect(1.0)
    plt.tight_layout()
        
    # Vergelijk de stijghoogte op afstand
    ztop = np.linspace(0, 80, 81) - 0.001j
    zbot = ztop - 0.999j * H
    
    wtop = get_w(ztop)
    wbot = get_w(zbot)
    
    clr=next(clrs)
    ax1.plot(ztop.real[1:], -wtop.real[1:], '-',  color=clr, label=f'top, H={H}')
    ax1.plot(zbot.real[1::5], -wbot.real[1::5], '.-', color=clr, label=f'bot, H={H}')
    
ax1.legend()

print(phinf)
print(phiLm)

fig.savefig(os.path.join('images', "sections"))
fig1.savefig(os.path.join('images', "tob_bot_potential"))

plt.show()

fig.savefig(os.path.join('images', f"sections_{case}"))
fig1.savefig(os.path.join('images', f"tob_bot_potential_{case}"))

plt.show()

# %%
