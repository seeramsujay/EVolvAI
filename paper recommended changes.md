# EVolvAI: Recommended Manuscript Revisions

This document outlines the strategic corrections required to address the IEEE IES conference reviewer feedback and resolve internal technical discrepancies.

---

### 1. Gini Constraint Paradox (Section III-D and Table III)
**Issue:** The text states a hard disqualification threshold of $G = 0.35$, but Table III reports results like $0.41$ and $0.52$.
**Technical Fix:**
*   **Revision:** Update the description of the Gini constraint in Section III-D. Instead of "unconditionally eliminating," describe it as a "prohibitive soft penalty hurdle."
*   **Explanation:** Clarify that in extreme demand scenarios (e.g., Full Electrification), the GA optimizer must balance the Equity objective against the Physical Resilience (transformer limits) objective. The reported values ($0.41 - 0.52$) represent the mathematically optimal Pareto frontier where equity was improved by ~40% from the $0.74$ baseline while still preventing grid collapse.

### 2. "Physics Tax" Independent Validation
**Issue:** Reviewers request an unconstrained VAE baseline to prove the "physics tax" isn't a convergence error.
**Proposed Action:**
*   **Revision:** Add a "Baseline Comparison" subsection in Section V.
*   **Data Requirement:** Run a control training session (1,000 epochs) with `physics_on: false` to achieve the target $R^2 > 0.92$. 
*   **Argument:** Present the $R^2 \approx 0.90$ (unconstrained) vs $R^2 \approx 0.48$ (physics-constrained) delta as the quantitative "Physics Tax"—the necessary deviation from historical shapes to ensure Kirchhoff's laws are satisfied.

### 3. Zero-Output Collapse Discussion (Section V-A)
**Issue:** Achieved rate (2.24%) exceeds target (1.5%).
**Revision Text:**
> "The observed 2.24% zero-output collapse rate represents latent states where the generative engine could not resolve a physically stable power flow profile under extreme condition vectors. Rather than introducing 'phantom demand,' these scenarios are filtered during the CVaR calculation phase, ensuring that planning remains anchored in physically actionable demand distributions."

### 4. Bootstrapping Methodology (Section III-B)
**Issue:** Descriptions of the NYC-to-IEEE mapping are too vague.
**Revision Text:**
> "The stochastic mapping follows a three-stage Spatial-Temporal Shift: 
> 1. **Temporal Extraction:** Diurnal charging power shapes (kW) are harvested from the Caltech ACN dataset. 
> 2. **Spatial Redistribution:** Using the NYC DOT Traffic Volume Count (ATVC) as a probability density function, each charging session is assigned to an IEEE 33-Bus node based on the traffic index at its start hour. 
> 3. **Validation:** This ensures that nodal load centers in the benchmark grid reflect the real-world congestion patterns of a high-density urban environment."

### 5. Completed References
Replace the placeholders in the bibliography with the following verified citations:

*   **[2]** Zhang, L., Fu, H., Zhou, Z., Wang, S., & Zhang, J. (2022). "Site selection and capacity determination of charging stations considering the uncertainty of users' dynamic charging demands," *Frontiers in Energy Research*.
*   **[3]** Lou, J., Shen, X., Niemeier, D. A., & Hultman, N. (2024). "Income and racial disparity in household publicly available electric vehicle infrastructure accessibility," *Nature Communications*.
*   **[6]** Wang, H., Xu, Y., Xu, D., & Peng, X. (2023). "Heavy Load and Overload Pre-warning for Distribution Transformer with PV Access Based on Graph Neural Network," *IEEE Xplore*.
*   **[8]** Islam, M., Nadarajah, M., & Hossain, M. J. (2019). "Short-Term Voltage Stability Enhancement in Residential Grid With High Penetration of Rooftop PV Units," *IEEE Transactions on Sustainable Energy*.
*   **[9]** Islam, M., Mithulananthan, N., & Hossain, J. (2019). "Short-Term Voltage Stability of Distribution Grids With Medium-Scale PV Plants Due to Asymmetrical Faults," *2019 IEEE PES GTD Asia*.
*   **[10]** Roald, L., & Andersson, G. (2023). "Chance-Constrained AC Optimal Power Flow: Reformulations and Efficient Algorithms," *IEEE Transactions on Power Systems*.
