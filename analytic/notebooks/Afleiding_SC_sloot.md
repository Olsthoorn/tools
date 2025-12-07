# Afleiding van de SC-transformatie voor de halve sloot (correcte hoeken)

We beschouwen de polyline met de punten:

$$
A = \infty + 0i, \quad 
B = b + 0i, \quad 
C = b - c i, \quad 
D = 0 - c i, \quad 
E = 0 - e i, \quad 
F = \infty - e i
$$

met $b, c, e > 0$ en $e > c$. De hoeken bij de knikpunten zijn:

$$
k_B = \frac{1}{2}, \quad 
k_C = -\frac{1}{2}, \quad 
k_D = \frac{1}{2}, \quad 
k_E = \frac{1}{2}
$$

en de punten op oneindig ($A$ en $F$) kunnen worden weggelaten.

---

## 1. Basis-integral van de SC-transformatie

Voor de overblijvende punten:

$$
z = \int_B^D (s - B)^{-1/2} (s - C)^{1/2} (s - D)^{-1/2} (s - E)^{-1/2} \, ds
$$

Let op: de factor \((s-C)^{1/2}\) staat nu in de **teller** vanwege de negatieve hoek bij \(C\).

---

## 2. Substitutie naar elliptische integralen

We maken de standaard-substitutie:

$$
s = B + (D-B) \, \text{sn}^2(u,m)
$$

waar $\text{sn}(u,m)$ de Jacobi elliptische sinus is en $m$ de modulus die afhangt van $b$, $c$ en $e$.

De integraal wordt dan:

$$
z = \int_0^K \frac{\sqrt{s(u)-C} \, 2 \sqrt{D-B} \, du}{\sqrt{(s(u)-B)(s(u)-D)(s(u)-E)}}
$$

met $K = \text{ellipk}(m)$, de kwartperiode van de elliptische integraal.

---

## 3. Bepaling van de modulair parameter $m$

De parameter $m$ wordt gekozen zodat de afbeelding precies van $B \to D$ en $C \to E$ loopt. Analytisch geldt (aanpassing door de teller-factor):

$$
m = \text{(afhankelijk van b, c, e en de factor } \sqrt{D-B}/\sqrt{s-C}\text{)}
$$

De exacte uitdrukking wordt bepaald door de standaardreductie van de SC-integraal naar de normale vorm van de elliptische integraal.

---

## 4. Inverse-transformatie

De inverse, van $z$ terug naar $s$, is:

$$
s = B + (D-B) \, \text{sn}^2\Bigg(\frac{z}{\int_0^K \frac{\sqrt{s(u)-C} \, 2 \sqrt{D-B} \, du}{\sqrt{(s(u)-B)(s(u)-D)(s(u)-E)}}}, m \Bigg)
$$

Hiermee kan elk punt langs de polyline worden getransformeerd naar de halve-sloot-ruimte met symmetrie-as $x=0$.

---

## 5. Rotatie en verschuiving

Indien gewenst kunnen we na de transformatie de $z$-ruimte nog manipuleren om:

- Punt $B$ naar $-1$ te verplaatsen,
- Punt $D$ naar $+1$ te verplaatsen,
- Een rotatie toe te passen door vermenigvuldiging met $-i$,
- De stroming van oneindig naar de halve sloot te oriënteren.

Dit levert een **analytische, exact renderbare kaart van de stroming** vanaf oneindig naar de halve sloot.


Python implementatie

import numpy as np
from scipy import integrate, special

# --- polyline parameters ---
b = 1.0   # breedte halve sloot
c = 0.5   # diepte sloot
e = 0.8   # laagdikte
# punten in s-space
B, C, D, E = b, b - 1j*c, 0 - 1j*c, 0 - 1j*e

# exponenten volgens hoek: k_B, k_C, k_D, k_E
k_B, k_C, k_D, k_E = 0.5, -0.5, 0.5, 0.5

# --- functie voor integrand ---
def integrand(s):
    return (s-B)**(-k_B) * (s-C)**(-k_C) * (s-D)**(-k_D) * (s-E)**(-k_E)

# --- numerieke integratie van B -> D langs het polyline ---
# we gebruiken een parametrische benadering: s(t) lineair van B naar D
def s_line(t):
    # t in [0,1] over B -> D
    return B + t*(D-B)

def ds_dt(t):
    return D-B

def integrand_param(t):
    return integrand(s_line(t)) * ds_dt(t)

# numerieke integraal
z_BD, err = integrate.quad(lambda t: np.abs(integrand_param(t)), 0, 1)
print("Numerieke waarde van |z_BD|:", z_BD)

# optioneel: voor meerdere punten langs polyline
npts = 50
tvals = np.linspace(0,1,npts)
zvals = np.array([integrate.quad(lambda t: np.abs(integrand_param(t)), 0, t)[0] for t in tvals])

# plotten van z
