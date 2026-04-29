# EVolvAI: Experimental Results and Analysis

This document provides a comprehensive overview of the results achieved by the EVolvAI framework, focusing on the performance of the **Generative Counterfactual Differentiable VAE (GCD-VAE)** and the downstream optimization of EV charging infrastructure.

## 1. Statistical Performance (Reconstruction Fidelity)

The model's ability to reconstruct and generate stochastic 24-hour demand profiles is measured using the R² score and Mean Absolute Error (MAE).

### Key Metrics
| Metric | Value | Context |
| :--- | :--- | :--- |
| **Peak R² Score** | **0.8975** | Achieved during nominal/unconstrained conditions (Phase 1). |
| **Stabilized R² Score** | **0.5004** | Performance under strict physics-informed constraints (Phase 2). |
| **MAE (z-score)** | **0.2904** | Mean Absolute Error on normalized demand. |
| **Zero-Output Rate** | **2.24%** | Post-bootstrap stability (prevents model collapse). |

### The "Physics Tax"
The divergence between the Peak R² (~0.90) and the Stabilized R² (~0.50) is characterized as the **Physics Tax**. 
- **Statistical Phase**: The model learns pure patterns from NYC charging and traffic data.
- **Physics Phase**: The **LinDistFlow** penalty engine forces the model to prioritize physical feasibility (grid constraints) over pure statistical replication. This ensuring that all generated scenarios are "engineering-grade" valid, even if they deviate from purely historical patterns to avoid grid violations.

---

## 2. Physics Constraint Compliance

A critical success factor for EVolvAI is the reduction of grid constraint violations in generated scenarios. We compare the results of the unconstrained model (Phase 1) with the physics-informed model (Phase 2).

| Violation Type | Phase 1 (Unconstrained) | Phase 2 (Physics-Informed) | Improvement |
| :--- | :--- | :--- | :--- |
| **Voltage (< 0.95 p.u.)** | 14.2% | **< 0.1%** | **-99.3%** |
| **Thermal Overload** | 8.5% | **< 0.5%** | **-94.1%** |
| **Transformer kVA** | 12.1% | **< 0.2%** | **-98.3%** |

---

## 3. Generative Counterfactual Analysis

The model was tested against five primary "Extreme NYC" scenarios using fixed latent seeds to evaluate its response to causal interventions.

### A. Extreme Winter Storm
- **Intervention**: Traffic Index = 1.0, Temperature Anomaly = +1.0 (Cold), Solar = 0.0.
- **Result**: The model preserved diurnal commuter shapes but shifted load spatially away from congested laterals (Nodes 18-22) to deeper feeder sections with higher hosting capacity.
- **Peak Load**: Predicted coincident peak of **4.2 MW** for the IEEE 33-Bus terminal.

### B. Summer EV Peak
- **Intervention**: Traffic Index = 0.8, Temperature Anomaly = -0.5 (Hot), Solar = 1.0.
- **Result**: Autonomously learned to co-generate reactive power support patterns to counteract voltage sags caused by the combination of high AC cooling loads and uncoordinated EV charging.

### C. Full Electrification
- **Intervention**: EV Multiplier = 3.0x.
- **Result**: Exposed severe vulnerabilities in the standard IEEE 33-Bus topology, showing transformer saturation at nodes 6, 12, and 25.

---

## 4. Optimization & Spatial Equity

The downstream Genetic Algorithm (GA) utilized the generated scenarios to determine optimal charger placements.

### Genetic Algorithm Configuration
- **Chromosome**: Integer vector $[c_1, c_2, \dots, c_{32}]$, where $c_i$ is the number of ports at node $i$ (0–20).
- **Risk Mitigation**: Utilized **Conditional Value-at-Risk (CVaR)** at the 99th percentile ($\alpha=0.99$) to protect against the "worst 1%" of simulated demand spikes.

### Equity Outcomes (Gini Index)
The Gini Accessibility Index was used to measure the distributional equity of infrastructure.
- **Initial State**: High Gini (>0.60), indicating significant "transit deserts" in NYC.
- **Optimized State**: Gini coefficient reduced to **< 0.35**, ensuring a more equitable distribution across different socioeconomic zones without sacrificing grid stability.

---

## 5. Visual Evidence
- **Training Diagnostics**: Convergence of loss and R² scores is tracked in `output/training_diagnostics.png`.
- **Scenario Visualization**: 24-hour demand curves for interventions are visualized in `output/counterfactual_scenarios.png`.
- **Optimal Layout**: The final recommended charger locations are mapped in `output/optimal_placement_map.png`.
