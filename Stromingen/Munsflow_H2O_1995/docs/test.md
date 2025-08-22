Absolutely—you can “invert” it that way: generate a synthetic van Genuchten–Mualem (vG–M) dataset and fit Brooks–Corey (BC) parameters to it. That’s often the cleanest way to ensure apples-to-apples behavior in your model.

## Practical recipe (robust and fast)

1. **Generate vG–M synthetic data**

   * Pick $(\theta_s,\theta_r,\alpha,n,m=1-1/n,K_s,\ell)$.
   * Create a suction grid $\psi$ (e.g. log-spaced from \~1 cm to, say, 10⁵ cm).
   * Compute $S_{e,\mathrm{VG}}(\psi)=[1+(\alpha\psi)^n]^{-m}$.
   * Compute $\theta_{\mathrm{VG}}=\theta_r+S_e(\theta_s-\theta_r)$.
   * Compute $K_{\mathrm{VG}}(\psi)$ via Mualem (or your chosen conductivity model).

2. **Fit BC retention first (recommended)**

   * BC retention (for $\psi\ge\psi_b$): $S_{e,\mathrm{BC}}=(\psi_b/\psi)^{\lambda}$; and $S_e=1$ for $\psi<\psi_b$.
   * Unknowns: $\psi_b>0$, $\lambda>0$. (Keep $\theta_s,\theta_r$ fixed to the vG values for a clean comparison.)
   * **Objective** (robust): minimize $\sum w_i\,[\log S_{e,\mathrm{VG}}(\psi_i)-\log S_{e,\mathrm{BC}}(\psi_i)]^2$ over $\psi\ge\psi_b$.

     * Log on $S_e$ emphasizes the dry tail where differences matter.
     * Use bounds and good initials: $\psi_b^{(0)}\!=\!1/\alpha,\; \lambda^{(0)}\!=\!n-1$.

3. **Then fit conductivity**

   * Choose your BC conductivity closure (Mualem-BC or Burdine-BC). Each gives a closed-form $K_{\mathrm{BC}}(S_e;\lambda,\ell)$ (with $K_s$ common).

   * Either:

     * **(a) Fix $\psi_b,\lambda$** from retention fit, fit only $\ell$ (and optionally a small scale tweak in $\psi_b$) to $\log K$; or
     * \*\*(b) Do a **joint fit** on $\theta$ and $K$ with a weighted objective:

       $$
       J=\sum w_\theta[\theta_{\mathrm{VG}}-\theta_{\mathrm{BC}}]^2+\sum w_K[\log K_{\mathrm{VG}}-\log K_{\mathrm{BC}}]^2.
       $$

       Start from $\psi_b=1/\alpha,\;\lambda=n-1,\;\ell=0.5$.

   * Fit $K$ in **log space** and **exclude/low-weight the near-saturated range** (where tiny head errors create large $K$ swings).

4. **Region selection / masking**

   * For BC retention, **only include $\psi\ge\psi_b$** in the objective (the dry branch).
   * For conductivity, ignore points where $S_e\to 1$ (or cap weights above $S_e>0.95$).

5. **Diagnostics**

   * Plot $S_e(\psi)$ and $K(\psi)$ on log scales (ψ on log axis) to verify slopes match in the dry tail.
   * Check that your fitted $\lambda\approx n-1$ and $\psi_b\approx 1/\alpha$; small deviations are expected because VG has a smooth “air-entry” while BC is piecewise.

## Tips that prevent headaches

* **Identifiability:** If you try to fit $(\theta_s,\theta_r,\psi_b,\lambda)$ all at once from synthetic vG, the problem can become sloppy. Fix $\theta_s,\theta_r$ to vG’s values unless you intentionally want differences.
* **Weighting:** Set $w_K$ so the K-fit doesn’t dominate the retention fit (e.g., normalize by the number of points and use log-space for K).
* **Initialization:** The asymptotic mapping is a great start: $n=\lambda+1$, $\alpha\approx 1/\psi_b$.
* **Purpose-driven fit:** If your model is sensitive to mid-range moistures (root-zone), weight that ψ-range more.

This workflow will give you BC parameters that track your vG curves as closely as you like—both for $\theta(\psi)$ and $K(\psi)$—without relying on hand-wavy mappings. If you later want, I can draft a compact Python least-squares snippet that drops right into your current vG code.
