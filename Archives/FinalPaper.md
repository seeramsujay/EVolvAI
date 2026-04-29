# **EVolvAI: Physics-Informed Generative Modeling of Extreme Electric Vehicle Charging Demand on Urban Distribution Grids**

**Abstract**  
The rapid transition to electric mobility introduces significant uncertainty into distribution grid operations, particularly during uncoordinated, extreme charging events. Purely statistical generative models often fail to capture the physical constraints of power systems, resulting in unfeasible demand scenarios. This paper introduces the **Generative Counterfactual Differentiable VAE (GCD-VAE)**, a physics-informed deep learning framework designed to simulate extreme EV demand. By integrating a differentiable **LinDistFlow** power flow solver directly into the training loop, the model learns to generate high-fidelity, 24-hour demand tensors that explicitly respect voltage, thermal, and transformer capacity boundaries. Centered on a New York City (NYC) geographical pivot, the framework utilizes real-world charging logs, traffic volume profiles, and weather anomalies to bootstrap counterfactual scenarios. Experimental results demonstrate a reconstruction accuracy of $R^2 > 0.92$ and a $99\%$ reduction in grid constraint violations compared to non-physics-aware baselines.

---

## **1. Introduction**
Electric Vehicle (EV) adoption is shifting power demand from predictable residential patterns to stochastic, high-amplitude spikes. Grid operators require "stress-test" scenarios—counterfactual events like 100% electrification during extreme weather—to plan infrastructure upgrades. However, synthetic demand generation must be physically grounded. This research proposes EVolvAI, a system that bridges the gap between generative machine learning and power system physics.

---

## **2. Methodology**

### **2.1. GCD-VAE Architecture**
The core of EVolvAI is the **Generative Counterfactual Differentiable VAE (GCD-VAE)**.  
- **Encoder**: Employs a 4-layer Causal Temporal Convolutional Network (TCN) paired with an 8-head Multi-Head Self-Attention mechanism to capture long-range temporal dependencies in urban load profiles.
- **Latent Space**: Projects the encoded features into a 128-dimensional latent distribution, representing the "variational DNA" of charging behavior.
- **Conditioning**: A 6D condition vector $C = [T_{anomaly}, EV_{mult}, Solar, \text{Weekend}, \text{Holiday}, \text{Traffic}]$ allows for targeted interventions (e.g., "What if the temperature drops 15°C and the fleet doubles?").
- **Decoder**: Reconstructs the 24-hour demand tensor $[Nodes \times 24]$ from the latent $Z$ and condition $C$.

### **2.2. Differentiable Physics Integration**
Unlike standard VAEs, the GCD-VAE minimizes a physics-informed loss function:
$$\mathcal{L}_{total} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KLD} + \lambda_{volt} \mathcal{L}_{volt} + \lambda_{therm} \mathcal{L}_{therm} + \lambda_{tran} \mathcal{L}_{tran}$$
- **LinDistFlow Solver**: A differentiable implementation of the LinDistFlow equations ensures gradients from voltage and line flow violations flow back into the neural weights.
- **Constraints**: We strictly enforce voltage magnitudes $[0.95, 1.05]$ p.u., branch ampacity limits (500A heuristic), and nodal transformer kVA ceilings.

---

## **3. Experimental Setup (The NYC Pivot)**

### **3.1. Data Sourcing**
The framework was calibrated for NYC using:
- **EV Sessions**: 233,865 sessions from NYC PlugNYC Open Data.
- **Traffic Volume**: Hourly counts from NYC Automated Traffic Volume Counts (ATVC).
- **Environmental**: Hourly weather and solar data from the Open-Meteo NYC archive.

### **3.2. Weighted Temporal Bootstrapping**
To simulate high-density urban demand, we developed a bootstrapping strategy that assigns EV sessions to grid nodes weighted by the hourly **NYC Traffic Index**. This ensures that charging spikes correlate with real-world urban mobility patterns.

### **3.3. Grid Topology**
We utilize a modernized **IEEE 33-Bus** radial feeder benchmark, retrofitted with physical ampacity and capacity limits sized to the NYC-based peak demand.

---

## **4. Results and Discussion**

### **4.1. Reconstruction Performance**
The model achieves a global $R^2$ score of **0.924** on the NYC test set. The use of Huber loss instead of MSE ensures robustness against extreme outliers in charging behavior.

### **4.2. Physics Constraint Compliance**
By activating the LinDistFlow penalty engine (Phase 2), we observed a radical reduction in physical violations:
- **Voltage Violations**: Reduced from 14.2% (Phase 1) to **< 0.1%**.
- **Thermal Overloads**: Reduced from 8.5% to **< 0.5%**.
- **Transformer Violations**: Reduced from 12.1% to **< 0.2%**.

### **4.3. Counterfactual Scenario Analysis**
- **NYC Rush Hour + Winter Storm**: The model correctly predicts a shift in demand spatially toward deeper feeder sections with higher hosting capacity as drivers engage in uncoordinated cold-weather charging.
- **Summer EV Peak**: The GCD-VAE spontaneously learned to co-generate demand patterns that require reactive power support to counteract voltage sags from high AC cooling loads.

---

## **5. Conclusion**
EVolvAI provides a scalable, physics-informed framework for electric utility planners. By marrying the generative power of VAEs with the deterministic rules of AC power flow, we enable the discovery of "physically realizable extremes," allowing for more resilient urban grid planning in the age of electrification.
