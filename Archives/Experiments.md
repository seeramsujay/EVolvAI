# Experiment Setup: NYC-Pivoted Physics-Informed GCD-VAE

This document details the experimental configuration for the EVolvAI generative pipeline, specifically focused on the New York City (NYC) geographical pivot.

## 1. Geographical Context & Data Sourcing
**Objective**: Simulate extreme EV demand on distribution grids within the NYC metropolitan area.

| Dataset | Source | Temporal Range | Description |
| :--- | :--- | :--- | :--- |
| **EV Charging** | NYC PlugNYC | 2021-07-31 – 2026-04-13 | 233,865 public charging sessions. |
| **Traffic Flow** | NYC Open Data (ATVC) | 2021-01-01 – 2026-04-15 | Hourly vehicle volume across all boroughs. |
| **Weather** | Open-Meteo Archive | 2021-07-31 – 2026-04-13 | Hourly Temp (C), Precip (mm), and Solar availability. |

## 2. Model Architecture: GCD-VAE
The **Generative Counterfactual Differentiable VAE (GCD-VAE)** utilizes a Causal TCN-Attention architecture.

- **Encoder**: 
  - 4-layer Causal Temporal Convolutional Network (TCN).
  - 8-head Multi-Head Self-Attention layer.
  - Latent Dimension: 128.
- **Decoder**:
  - Concatenation of Latent Vector ($Z$) and Condition Vector ($C$).
  - 3-layer TCN for high-fidelity reconstruction.
- **Condition Vector ($C$)**: 6-dimensional [Temp Anomaly, EV Multiplier, Solar, Weekend, Holiday, Traffic Index].

## 3. Physics Integration (LinDistFlow)
The model is constrained by an IEEE 33-Bus radial feeder topology located at the "terminal" of the NYC simulation. 
- **Solver**: Differentiable LinDistFlow (V² formulation).
- **Constraints**:
  - **Voltage**: $[0.95, 1.05]$ p.u.
  - **Thermal**: Current limits per branch based on branch ampacity (500A heuristic).
  - **Capacity**: Nodal transformer limits (kVA) scaled to baseline peak.

## 4. Hyperparameters & Training Routine
Training is conducted in two distinct phases to ensure stable convergence of the physics engine.

- **Phase 1: Warm-up (Epochs 1–50)**:
  - Physics penalties disabled ($\lambda = 0$).
  - Focus: Latent space regularization and statistical reconstruction.
  - KLD Weight: Linear annealing from $0 \to 0.01$.
- **Phase 2: Physics-Aware (Epochs 51–200)**:
  - Physics penalties activated.
  - $\lambda_{voltage} = 1000.0$, $\lambda_{thermal} = 500.0$, $\lambda_{transformer} = 800.0$.
  - Adaptive KLD weight up to $0.1$.

## 5. Bootstrapping Strategy
To overcome data scarcity for extreme scenarios, we utilize a **Weighted Temporal Bootstrap**:
1. Sample 5,000 synthetic days from the 233,865 NYC session bank.
2. Distribute sessions across 32 IEEE-33 nodes.
3. Probability of node assignment is weighted by the **NYC ATVC Traffic Index** for that specific hour.
4. Result: 24-hour demand tensors $[32, 24]$ for each of the 5,000 training samples.
