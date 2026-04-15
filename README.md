# EVolvAI: Physics-Informed Generative EV Demand Pipeline

**EVolvAI** is a research-grade framework for modeling and optimizing electric vehicle (EV) charging infrastructure under extreme grid constraints. By combining a Physics-Informed VAE (GCD-VAE) with a Genetic Algorithm risk engine, EVolvAI generates physically valid counterfactual demand scenarios and identifies optimal charger placements to minimize grid reliability risks.

## Project Structure

```text
.
├── generative_core/      # GCD-VAE architecture and physics-informed training
├── data_pipeline/        # Data ingestion, weather fetching, and bootstrapping
├── risk_engine/          # Genetic Algorithm for optimal charger placement
├── geospatial_dashboard/ # Streamlit visualization for demand and grid risk
├── scripts/              # Utility scripts for quality checks and builds
├── Archives/             # Research papers, reports, and methodology
├── data/                 # Raw and processed datasets
└── output/               # Model checkpoints and generated scenarios
```

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
python data_pipeline/preprocess.py --synthetic
python data_pipeline/bootstrap.py --scenarios 5000
```

### 3. Training & Inference
The primary entry point is the **`Latest_Training.ipynb`** notebook, optimized for Google Colab.

To run via CLI:
```bash
python run.py train --epochs 500
python run.py generate
```

### 4. Optimize Placement
```bash
python run.py optimize
```

### 5. Visualize
```bash
streamlit run geospatial_dashboard/dashboard.py
```

## Publication Context
This repository contains the source code for the paper: *"Physics-Informed Generative Modeling of Extreme EV Demand on Distribution Grids."* focusing on a 40.7°N NYC-based case study on the IEEE 33-Bus system.

---
**License**: MIT  
**Research Context**: Advanced Agentic Coding / Power Systems Engineering.
