# Generative Core (GCD-VAE)

The `generative_core/` folder contains the architecture and training logic for the **Generative Counterfactual Differentiable VAE (GCD-VAE)**.

## Architecture

The model is a Causal TCN-Attention VAE designed to handle multivariate temporal demand profiles.

- **`models.py`**: Defines the `GenerativeCounterfactualVAE` with its temporal convolutional encoder and decoder.
- **`physics_loss.py`**: Implementation of the **Differentiable LinDistFlow** loss. It computes penalties for voltage sags and thermal overloads that can be backpropagated to the latent space.
- **`train.py`**: The modular training engine support annealing for both KLD weights (posterior collapse prevention) and physics lambdas (curriculum learning).
- **`generate.py`**: Inference engine for creating counterfactual scenarios from trained checkpoints.
- **`data_loader.py`**: Optimized PyTorch DataLoader with on-the-fly normalization and sequence creation.

## Design Principle

Unlike standard VAEs, the GCD-VAE is "physics-informed." The loss function includes term $\mathcal{L}_{physics}$ which forces the generated demand to stay within the engineering bounds of the IEEE-33 bus system.
