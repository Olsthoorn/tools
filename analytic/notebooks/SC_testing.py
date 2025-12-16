#!/usr/bin/env python
# coding: utf-8

# # Schartz-Christoffel for 6 points case Ditch with rectangular profile
# 
# From chatGPT
# 
# Sketch of the cross section
# 
#        B-----------------A
#        |
# D ---- C
# |
# E -----------------------F
# 
# We willen een Schwarz–Christoffel (SC) mapping (f) van de bovenhalf-vlakte (\mathbb{H}) (variabele (\zeta)) naar deze veelhoek in het (z)-vlak.
# 
# ---
# De uiteindeijke transformatie bestaat it een standaard $\sin$ en $\arcsin$ transformatie gevolgd door de SC-transformatie. We gaan van het $\Omega$-vlak via het $\zeta_0$-vlak naar het $\zeta$-vlak en vandaar met de Scharz-Cristoffel transformatie naar het $w$-vlak en tenslotte naar het $z$-vlak, dat de werkelijkheid voorstelt.
# 
# **Stap 1 — bepaal de binnenhoeken $\alpha_k$**
# **Stap 2 — zet de prevertices en onbekenden**
# 
# We hebben 4 punten (B, C, D en E). Punten op oneindig (A en F) kunnen we weglaten.
# De transformatie van $\zeta$-plane naar $w$-plane levert een gelijkvormige figuur op. We moeten ervoor zorgen dag de verhoudingen van de lengtes in orde zijn.
# 
# Als we de punten $w_B$, $w_C$, $w_D$ en $w_E$ hebben kunnen we die mappen op de werkelijke punten $z_B$, $z_C$, $z_D$ en $z_$E. Dit hoeft maar met twee van de punten te gebeuren. We hebben dan de mapping van zeta naar de werkelijkheid
# 
# De gehele conforme transformatie start in het $\Omega$-vlak waarin $\Psi$ de horizontale stroomfunctie-lijnen zijn (imaginaire waarden) en $\Phi$ de verticale potentiaallijnen. Voor het omega vlak kunnen we die direct aanmaken.
# 
# We kiezen dat de ondeste lijn $\Psi=0$ en de bovenste $\Psi=Q$.
# 
# Deze $\Omega$ transformeren we naar het zeta vlak waar de stroomlijnen zijn platgeslagen. We doen dit door eerst
# Omega 90 graden linksom te kantelen met $\pi$ te vermendigvuligen en door $Q$ te delen en $1/2$ af te trekken. De twee uiterste stroomfunctielijnen zijn nu twee verticale lijnen op resp. $\zeta_0=\pm \pi/2$
# 
# **Stap 3 — lengtevoorwaarden (parameterprobleem)**
# 
# De SC-transformatie van $\zeta$ naar $w$ zet punten $x_P$ op de reële $\zeta$-as ($x$-as) om naar een polygon met dezelfde vorm als de oorspronkelijke doorsnede (figuur). We zullen de punten xP optimaliseren zodanig dat de verhouding van de lengtes van de figuur hetzelfde zijn als in de werkelijkheid. In de SC transformatie liggen de hoeken tussen de ribben van het polygon vast. Het gaat dan alleen nog om de plek van de punten $x_P$ op de reële $\zeta$_as. Omdat we alleen de verhouging van de riblengte goed moeten krijgen, is de eerste riblengte om het even. Dat betekent dat de de eerste twee punten vrij kunnen kiezen. De litearatuur zegt dat we drie punten vrij kunnen kiezen, omdat de vorm van een driehoek uitsluitend wordt bepaald door zijn drie hoeken. Punet op oneindig mogen weggelaten worden.
# 
# **Stap 5 — verkrijg K en C en evalueer mapping**
# 
# Als de prevertices ($x_P$) gevonden zijn dan bepalen we de factor $P$ en de constante $Q$ die het $w$-vlak op het $z$-vlak (de werkelijkheid projecteert). Deze twee parameters zijn zijn eenduidig te bepalen door twee overeenkomstige punten op elkaar te mappen.
# 
# **Stap 6 — terugtransformeer (Möbius & rotatie)**
# 
# Ik zie niet hoe terugtransformatie mogelijk is. Maar we kunnen van $\Omega$ helemaal naar het $z$-vlak en dat is voldoende. We kunnen dan in het z-vlak de stroomlijnen en de potentiaal lijnen teken.
# 

# 
# ### SC-integrand voor jouw geval
# 
# De integrand (zonder constante (K)) wordt
# $$
# F(\omega);=;\prod_{k=1}^6 (\omega-a_k)^{\beta_k-1}
# \;=\;(\omega-x_A)^{0},(\omega-x_B)^{-1/2},(\omega-x_C)^{-1/2},(\omega-x_D)^{-1/2},(\omega-x_E)^{-1/2},(\omega-x_F)^{0}.
# $$
# De factoren met $x_A$ en $x_F$ (punten op oneindig) vallen weg en punten $x_B$ en $x_C$ kunnen vrij gekozen worden.
# 
# Dus praktisch slechts vier vierkantswortel-singulariteiten op de reële as — dat is numeriek goed behandelbaar.
# 
# De kaart is
# $$
# f(\zeta)=C + K\int^\zeta F(\omega),d\omega.
# $$

# In[11]:


import os, sys
import numpy as np   #noqa E402
import matplotlib.pyplot as plt  #noqa E402

# %%
k_i = np.array([0, -0.9])
eps = 1e-6
k_eff = max(0.0, max(k_i))
eps = 1e-8  # or whatever you trust
W = np.sqrt(np.log(1/((1-k_eff)*eps)) / (1-k_eff))
print(W)
# %%

0# In[ ]:


"""
Schwarz-Christoffel mapping: 6-vertex example (A=(inf,0), B=(b,0), C=(b,-c), D=(0,-c), E=(0,-e), F=(inf,0))

Dit is een uitvoerbaar, maar toch pedagogisch script: het legt de stappen uit, bouwt
het parameterprobleem op en geeft een numeriek startpunt. Het is een "working
skeleton" — in praktijk zul je voor stabiele convergentie soms startwaarden of
kleine implementatie-aanpassingen moeten doen (branch-keuze bij wortels, extra
precisie via mpmath, etc.).

Benodigd: numpy, scipy, mpmath, matplotlib (optioneel voor plots).

Belangrijk:
- Dit script behandelt de twee oneindige vertices door ze betas=1 te geven
  (geen singulariteit). In numerische praktijk is het vaak stabieler eerst een
  Möbius-transformatie uit te voeren die infinities in prevertex-ruimte naar
  eindige punten brengt; daarmee krijg je volledig eindige integratiegrenzen.
  De SC-Toolbox (MATLAB) automatiseert dat; in Python moet je het expliciet
  doen als je sterke convergentieproblemen ziet.

- Het doel hier: bouw een concrete routine die de prevertices op de reële as
  oplost zodanig dat de (geschaalde) SC-integralen de bekende z-veld
  zijlengtes (BC, CD, DE) reproduceren.

Auteur: (voorbeeld door ChatGPT voor Theo Olsthoorn)
"""


# subdiv line through points A and B in N steps

def subdiv(A, B, xP, N=100):
    alpha = np.linspace(0, 1, N)    
    ds = np.zeros_like(alpha) + np.inf
    for xi in xP:
        ds = np.fmin(ds, np.abs((A + alpha * (B - A)) - xi))
    
    ds *= np.abs(B - A) / np.sum(ds)
    
    fig, ax = plt.subplots()
    ax.plot(np.cumsum(ds) + A.real, ds)
    ax.plot(xP, np.zeros_like(xP), 'o')
    return ds
    
# ds = subdiv(A, B, xP, N=100)

        


erf# In[58]:


from scipy.interpolate import interp1d

def subdiv(A, B, xP, N=100, eps=1e-6, show=False):

    # parameter x ∈ [0,1]
    alpha = np.linspace(0, 1, 100)  # dense pre-grid
    
    # fysieke punten
    Z = A + alpha * (B - A)
    
    # distance-based weight
    w = np.min(np.abs(Z[:, None] - np.array(xP)[None, :]), axis=1) + eps
    
    # integrate 1/w
    s = np.cumsum(1/w)
    s /= s[-1]                       # normalize to [0,1]

    # invert s(α): given s_k → α_k
    
    alpha_new = np.interp(np.linspace(0, 1, N+1), s, alpha)

    # final points
    Z_new = A + alpha_new * (B - A)
    ds = np.abs(np.diff(Z_new))

    if show:
        fig, ax = plt.subplots()
        ax.plot(xP, np.zeros_like(xP), 'o')
        ax.plot(Z_new.real, Z_new.imag, '.')
        plt.show()

    return Z_new, ds

A, B = -4 + 0.1j, 4 + 0.1j
xP = [0, 2, 3, 4]


Z_new, ds = subdiv(A, B, xP, N=100, eps=1e-6)

fig, ax = plt.subplots()
Zm = 0.5 * (Z_new[:-1] + Z_new[1:])
ax.plot(Zm.real, Zm.imag, '.')
for zm, ds_ in zip(Zm, ds):
    ax.plot([zm.real, zm.real], [zm.imag, zm.imag + np.abs(ds_)], 'r-')
ax.plot(xP, np.zeros_like(xP), 'bo')
ax.plot()

plt.show()





# %%
