# %% imports
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown] We need explanation for the Green and Ampt model
# # Green and Ampt model
# The Green and Ampt model is a widely used approach in hydrology for simulating the
# infiltration of water into soil. It is based on the assumption that the infiltration rate
# is controlled by the capillary pressure head and the hydraulic conductivity of the soil.
# The model describes the relationship between the capillary pressure head, the soil
# moisture content, and the hydraulic conductivity, allowing for the calculation of the
# infiltration rate over time. The model is particularly useful for understanding the
# dynamics of water movement in unsaturated soils and is often applied in hydrological
# modeling and water resource management.
# The Green and Ampt model is characterized by its simplicity and effectiveness in
# representing the infiltration process. It assumes that the soil is homogeneous and
# isotropic, meaning that the properties of the soil do not vary with depth or direction.
# The model uses the capillary pressure head to determine the rate of infiltration, which is
# influenced by the soil's hydraulic conductivity and the moisture content. The model
# provides a framework for understanding how water moves through the soil, allowing for
# predictions of infiltration rates and the distribution of moisture within the soil profile.

# %% [markdown]
# ## Green and Ampt model using Brooks and Corey (1964) soil-moisture and head relationships
# The Green and Ampt model can be implemented using the soil-moisture and head
# relationships proposed by Brooks and Corey (1964). In this approach, the capillary
# pressure head is related to the soil moisture content through a power-law function.
# The key equations for the Green and Ampt model using Brooks and Corey relationships
# are as follows:
# 1. **Capillary pressure head**: The capillary pressure head $h$ is given by:
#    $$h = h_b \left( \frac{\theta - \theta_r}{\theta_s - \theta_r} \right)^{-\lambda}$$    
#    where:
#    - $h_b$ is the bubbling pressure head,
#    - $\theta$ is the volumetric water content,
#    - $\theta_r$ is the residual water content,
#    - $\theta_s$ is the saturated water content,
#    - $\lambda$ is a parameter that describes the shape of the soil moisture curve.
# 2. **Moisture content**: The moisture content $\theta$ is related to the capillary
#    pressure head by the equation:
#    $$\theta = \theta_r + (\theta_s - \theta_r) \left(1 - \left(\frac{h_b}{h}\right)^{\frac{1}{\lambda}}\right)$$
#    This equation describes how the moisture content changes with the capillary pressure head.
# 3. **Hydraulic conductivity**: The hydraulic conductivity $K$ is given by:
#    $$K = K_s \left(1 - \left(\frac{h_b}{h}\right)^{\frac{1}{\lambda}}\right)$$
#    This equation relates the hydraulic conductivity to the capillary pressure head and the
#    saturated hydraulic conductivity.

# %% [markdown]
# Brooks and Corey (1964)
# %%
# Reduced satuaration index # S = (theta - theta_r) / (theta_s - theta_r)
def S(theta, theta_r, theta_s):
    """Calculate the reduced saturation index S."""
    return (theta - theta_r) / (theta_s - theta_r)

def psi2Theta_BC(Psi, Psi_b, lambda_):
    """Calculate reduced saturation S from the capillary pressure head Psi."""
    
    Theta = (Psi_b / Psi) ** lambda_
    Theta[Psi <= Psi_b] = 1.0  # Saturation is 1 for Psi <= Psi_b
    return Theta

def psi2Theta_vG(Psi, alpha, M):
    """Calculate reduced saturation S from the capillary pressure head Psi using van Genuchten model."""
    N = 1 / (1 - M)
    
    Theta = (1 / (1 + (alpha * Psi) ** N)) **  M
    Theta[Psi <= 0.] = 1.0  # Saturation is 1 for Psi <= 0
    return Theta

def Theta2Psi_BC(Theta, Psi_b, lambda_):
    """Calculate capillary pressure head Psi from reduced saturation S using Brooks and Corey model."""
    Psi = Psi_b / (Theta ** (1 / lambda_))
    return Psi

def Theta2Psi_vG(Theta, alpha, M):
    """Calculate capillary pressure head Psi from reduced saturation S using van Genuchten model."""
    N = 1 / (1 - M)
    Psi = (Theta ** (-1 / M) - 1) ** (1 / N) / alpha
    return Psi

Psi_b, lambda_ = 41, 3.7
alpha, M = 0.0202, 0.87

Psi = np.logspace(1, np.log10(200), 31)
Theta_BC = psi2Theta_BC(Psi, Psi_b, lambda_)

Theta_vG = psi2Theta_vG(Psi, alpha, M)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Reduced Saturation Index vs. Capillary Pressure Head\nCharbeneau (2000), fig 4.4.3')
ax.set_ylabel('Capillary Pressure Head (Psi) [cm]')
ax.set_xlabel('Reduced Saturation Index (S)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_ylim(0, 200)

ax.plot(Theta_BC, Psi, marker='o', linestyle='-', color='b', label=fr"Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Theta_vG, Psi, marker='x', linestyle='--', color='r', label=fr"Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.legend(loc='upper right')
plt.show()

# %%
def Theta2krw_BC(Theta, lambda_):
    """Calculate relative permeability for water using Brooks and Corey model."""
    Theta[Theta > 1.0] = 1.0
    epsilon = 3. + 2. / lambda_
    return Theta ** epsilon

def Theta2krw_vG(Theta, M):
    """Calculate relative permeability for water using van Genuchten model."""
    Theta[Theta > 1.0] = 1.0
    return np.sqrt(Theta) * (1 - (1 - Theta ** (1/ M)) ** M) ** 2

def Theta2kra_BC(Theta, lambda_):
    """Calculate relative permeability for air using Brooks and Corey model."""
    Theta[Theta > 1.0] = 1.0
    return (1 - Theta) ** 2 * (1 - Theta ** (1. + 2. /lambda_))

def Theta2kra_vG(Theta, M):
    """Calculate relative permeability for air using van Genuchten model."""
    Theta[Theta > 1.0] = 1.0
    return (1 - np.sqrt(Theta)) * (1 - Theta ** (1 / M)) ** (2 * M)

# Parameters for the models
lambda_, M = 3.7, 0.87
Theta = np.linspace(0, 1.0, 51)[1:]

krw_bc = Theta2krw_BC(Theta=Theta, lambda_=lambda_)
krw_vg = Theta2krw_vG(Theta=Theta, M=M)
kra_bc = Theta2kra_BC(Theta=Theta, lambda_=lambda_)
kra_vg = Theta2kra_vG(Theta=Theta, M=M)

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_ylabel('relative hydraulic conductivity (k_rw) [cm/d]')
ax.set_xlabel('Reduced Saturation Index (Theta)')
ax.set_title('Relative hydraulic conductivity\nCharbeneau (2000), fig 4.4.4')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.plot(Theta, krw_bc, marker='o', linestyle='-', color='b', label=fr"krw, Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Theta, krw_vg, marker='x', linestyle='--', color='r', label=fr"krw, Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.plot(Theta, kra_bc, marker='s', linestyle=':', color='g', label=fr"kra,  Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Theta, kra_vg, marker='d', linestyle='-.', color='m', label=fr"kra,  Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.legend(loc='upper right')
plt.show()

# %% Psi as a function of Theta

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('Capillary Pressure Head vs. Reduced Saturation Index\nCharbeneau (2000), fig 4.4.3')
ax.set_ylabel('Capillary Pressure Head (Psi) [cm]')
ax.set_xlabel('Reduced Saturation Index (Theta)')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)  

Psi_BC = Theta2Psi_BC(Theta_BC, Psi_b, lambda_)
Psi_vG = Theta2Psi_vG(Theta_vG, alpha, M)

ax.plot(Theta_BC, Psi_BC, marker='o', linestyle='-', color='b', label=fr"Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Theta_vG, Psi_vG, marker='x', linestyle='--', color='r', label=fr"Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.legend(loc='upper right')
plt.show()


# %% krw and kra as a function of Psi

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('krw and kra vs capillary pressure\nCharbeneau (2000), fig 4.4.4')
ax.set_xlabel('Capillary Pressure Head (Psi) [cm]')
ax.set_ylabel('krw, kra [-]')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)  

Psi_BC = Theta2Psi_BC(Theta_BC, Psi_b, lambda_)
Psi_vG = Theta2Psi_vG(Theta_vG, alpha, M)

theta_bc = psi2Theta_BC(Psi_BC, Psi_b, lambda_)
theta_vG = psi2Theta_vG(Psi_vG, alpha, M)

krw_bc = Theta2krw_BC(Theta=theta_bc, lambda_=lambda_)
krw_vg = Theta2krw_vG(Theta=theta_vG, M=M)
kra_bc = Theta2kra_BC(Theta=theta_bc, lambda_=lambda_)
kra_vg = Theta2kra_vG(Theta=theta_vG, M=M)

ax.plot(Psi_BC, krw_bc, marker='o', linestyle='-', color='b', label=fr"krw, Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Psi_vG, krw_vg, marker='x', linestyle='--', color='r', label=fr"krw, Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.plot(Psi_BC, kra_bc, marker='s', linestyle='-', color='b', label=fr"kra, Brooks and Corey, fine sand, $\Psi_b$={Psi_b} cm, $\lambda$={lambda_}")

ax.plot(Psi_vG, kra_vg, marker='+', linestyle='--', color='r', label=fr"kra,  Van Genuchten, fine sand, $\alpha$={alpha}, $M$={M}")

ax.legend(loc='upper right')
plt.show()

# %%

def int_krw_dpsi_BC(psi_0, psi_z, psi_b, lambda_):
    """Integrate relative permeability for water over capillary pressure head between phi_0 and phi_z."""
    # psi_0 = np.max(psi_b, psi_0)  # Ensure psi_0 is not less than psi_b
    # psi_z = np.max(psi_b, psi_z)  # Ensure psi_z is not
    return psi_b  / (1 + 3 * lambda_) * (
        (psi_b / psi_0) ** (1 + 3 * lambda_) - (psi_b / psi_z) ** (1 + 3 * lambda_)
        )
    
def int_krw_dpsi_vG(Psi_0, Psi_1, M, alpha):
    """Integrate relative permeability for water over capillary pressure head between phi_0 and phi_z."""
    # Psi_0 = np.max(0, Psi_0)  # Ensure Psi_0 is not less than 0
    # Psi_1 = np.max(0, Psi_1)  # Ensure Psi_1 is not less than 0

    from scipy.integrate import quad
    
    def integrand(Psi, M, alpha):        
        N = 1 / (1 - M)
        Theta = (1  / (1 + (alpha * Psi) ** N)) ** M        
        return np.sqrt(Theta) * (1 - (1 - Theta ** (1/M)) ** M) ** 2        

    def krw_integral(Psi0, Psi1, M, alpha):        
        result, _ = quad(integrand, Psi0, Psi1, args=(M, alpha))
        return result

    return krw_integral(Psi_0, Psi_1, M, alpha)

# %% Velocity of the front dK(theta) / dtheta
theta_r = 0.10  # Residual water content
theta_s = 0.35  # Saturated water content
K_s = 0.02
   
def dK_dtheta_BC(theta, epsilon=3.5):
    """Calculate the derivative of relative permeability for water with respect to theta using Brooks and Corey model."""    
    K = K_s * ((theta - theta_r) / (theta_s - theta_r)) ** epsilon
    
    dK_dtheta = epsilon * K / (theta_s - theta_r) * ((theta - theta_r) / (theta_s - theta_r)) ** (epsilon - 1)
    
    return K, dK_dtheta
    
theta = np.linspace(theta_r, theta_s, 35)
K_bc, dK_dtheta_bc = dK_dtheta_BC(theta, epsilon=3.54)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
ax1.set_title('Derivative of Relative Permeability for Water with respect to Theta\nBrooks and Corey Model')

ax1.set_ylabel('dK/dTheta [-]')
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
ax1.legend(loc='upper right')
ax1.plot(theta, dK_dtheta_bc, linestyle='-', color='b', label=fr'dK/dTheta, Brooks and Corey, $\epsilon$={0.5}')

ax2.set_title('Permeability for Water vs. Theta')
ax1.set_ylabel('K_bc [-]')
ax2.set_xlabel('theta')
ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
ax2.plot(theta, K_bc, linestyle='-', color='r', label=fr'K_Brooks and Corey, $\epsilon$={0.5}')
ax2.legend(loc='upper right')
plt.show()



# %%

psi_0, psi_z, psi_b, M, lambda_, alpha = 0.001, 410, 41, 0.87, 3.7, 0.0202
print("int_krw_dpsi_BC = ", int_krw_dpsi_BC(psi_0, psi_z, psi_b, lambda_))
print("int_krw_dpsi_vG = ", int_krw_dpsi_vG(psi_0, psi_z, M, alpha))


# %%
