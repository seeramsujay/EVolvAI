# Data Pipeline

This directory contains the scripts and tools for fetching, cleaning, and preparing data for the EVolvAI generative model.

## Contents

- **`bootstrap.py`**: The primary data preparation script. It performs a **Weighted Temporal Bootstrap** on NYC charging sessions, distributing them across grid nodes based on hourly traffic volume.
- **`preprocess.py`**: Processes raw ACN-Data or NYC Charging CSVs into normalized training parquets.
- **`traffic_preprocess.py`**: Handles NYC Automated Traffic Volume Counts (ATVC), generating hourly traffic indices used as causal triggers in the VAE.
- **`fetch_weather.py`**: Automated fetcher for NYC historical weather (Temperature, Precipitation, Solar) matching the EV session timeframe.
- **`ieee33bus_data.py`**: Static topology and impedance data for the IEEE 33-Bus radial distribution network.
- **`physics_penalty_engine.py`**: Calculates grid constraint violations (Voltage, Thermal, Transformer) for use in testing and evaluation.

## Workflow

1.  **Raw Data**: Place raw CSVs in `data/raw/`.
2.  **Preprocess**: Run `python data_pipeline/preprocess.py` to generate the baseline training set.
3.  **Bootstrap**: Use `python data_pipeline/bootstrap.py` to expand the dataset into 5,000+ counterfactual scenarios for robust VAE training.
