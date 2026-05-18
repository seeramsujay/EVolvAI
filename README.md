# ⚡ EVolvAI: Physics-Informed Generative EV Demand Pipeline

EVolvAI is a research-grade cyber-physical framework designed to model and optimize electric vehicle (EV) charging infrastructure under extreme distribution grid constraints and severe climate anomalies. 

By combining a **Physics-Constrained Generative Counterfactual VAE (GCD-VAE)** with a **Multi-Objective Genetic Algorithm (GA) Risk Engine**, it generates physically valid charging demand scenarios and optimizes charger node distribution to mitigate grid collapse.

---

## 🏗️ System Pipeline & Architecture

The framework orchestrates a seamless four-step pipeline to transform climate/traffic anomalies into optimized grid infrastructure:

```
    [ Weather (Open-Meteo) ]      [ Traffic Index & ACN-Data ]
                │                              │
                └──────────────┬───────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   1. Data Pipeline & Bootstrap Preprocess    │
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   2. Generative Core: Attention-TCN GCD-VAE  │  ◄── [ gcvae_model.pt ]
        └──────────────────────┬───────────────────────┘
                               ▼ (NumPy Tensors: e.g. extreme_winter_storm.npy)
        ┌──────────────────────────────────────────────┐
        │   3. Genetic Algorithm (GA) Risk Engine      │  ◄── (IEEE 33-Bus Topology)
        └──────────────────────┬───────────────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │   4. Geospatial NYC Streamlit/FastAPI Map    │  ◄── (Gini Equity Index)
        └──────────────────────────────────────────────┘
```

---

## ⚡ Key Persuasion Points & Features

- **Physics-Constrained VAE (GCD-VAE)**: An Attention-TCN Variational Autoencoder that learns the temporal relationships between climate, traffic, and charger demands, maintaining physical conservation laws to prevent impossible demand counterfactuals.
- **Multi-Objective Genetic Algorithm**: Optimizes a vector of length 32 (ports per grid node on the IEEE 33-Bus standard) to balance CapEx, average wait times, transformer grid stress, and tail-risk (Conditional Value at Risk - CVaR).
- **Gini Social Equity Indexing**: Calculates the Gini Accessibility Index across local demographics to ensure charging infrastructure is distributed equitably and is not concentrated solely in affluent areas.
- **NYC Geospatial Dashboard**: Streamlit-driven interactive map which projects the abstract IEEE 33-Bus topology onto real New York City coordinates, overlaying real-world chargers (OpenChargeMap API) with the GA's optimized recommendations.
- **Independent Handoff Architecture**: The pipeline isolates PyTorch generative modeling and the Streamlit/FastAPI dashboard using standardized NumPy tensors (`.npy`) and JSON configs, allowing components to run independently.

---

## 🛠️ Environmental Constraints & Protocol Alignment

Aligned with the **ANTIGRAVITY Protocol**, the repository is optimized for speed and token-efficiency:
- **Compute Optimization**: Employs pre-trained model weights (`gcvae_model.pt`) and Colab-optimized workflows (`Latest_Training.ipynb`) to bypass heavy localized CPU training, running lightning-fast inferences on dual-core setups (Mac Air 2017 i5, 8GB RAM).
- **Data Footprint**: Relies on structured binary `.npy` tensors and lightweight SQLite/JSON stores for ultra-low latency data transfers.

---

## 🚀 Quick Start (60-Second Onboarding)

### 1. Install System Dependencies
Ensure Python 3.10+ is active:
```bash
pip install -r requirements.txt
```

### 2. Preprocess & Bootstrap Data
Generate the baseline synthetic grids and scenario matrices:
```bash
python data_pipeline/preprocess.py --synthetic
python data_pipeline/bootstrap.py --scenarios 5000
```

### 3. Run Generative Demand Inference
Generate the extreme weather counterfactual tensors using the pre-trained GCD-VAE:
```bash
python run.py generate
```

### 4. Execute GA Infrastructure Optimization
Find the optimal charger ports distribution to safeguard the IEEE 33-Bus grid:
```bash
python run.py optimize
```

### 5. Launch the Geospatial Dashboard
Deploy the interactive mapping visualization:
```bash
streamlit run geospatial_dashboard/dashboard.py
```

---

## 🔬 Scientific Validation
The methodology and empirical evaluations are documented in:
*   `PROJECT_OVERVIEW.md`: Full architectural deep-dive and scenario hyperparameters.
*   `results.md`: Complete tabular summary of GA fitness metrics and Gini equity quotients.
*   `working_process.md`: Detailed logs on model convergence and training loss profiles.

---

## 📄 License
Released under the MIT License. See `LICENSE` for details.
