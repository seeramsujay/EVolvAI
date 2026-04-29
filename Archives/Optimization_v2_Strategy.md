# Optimization Strategy v2: The "Green Light" Upgrades

This document details the refined architecture and loss functions proposed to break the $R^2 = 0.50$ ceiling and reach targets of **$0.89+$**.

---

## 🟢 1. Positional Encoding (Temporal Anchoring)
**Problem**: The `nn.MultiheadAttention` treats the 24-hour sequence as an "unordered bag" of features. It has no concept of whether a load spike is at 3:00 AM or 6:00 PM.

**Technical Spec**:
- **Module**: Implement a `PositionalEncoding` class using sine and cosine frequencies.
- **Injection Points**:
    1. **Encoder**: Inject PE into the TCN output before the `TransformerEncoder`.
    2. **Decoder**: Inject PE into the latent vector projection before it enters the `dec_tcn`.
- **Expected Impact**: Instantly "anchors" rush-hour spikes to clock-time, significantly boosting $R^2$ reconstruction fidelity.

## 🟢 2. Peak Demand Loss (Ceiling Breaker)
**Problem**: MSE and Huber losses are "safe"—they minimize broad errors by predicting the average, which results in smoothed-out peaks that fail to capture the physics of EV grid spikes.

**Technical Spec**:
- **Expression**: `peak_loss = F.mse_loss(recon.max(dim=-1)[0], x.max(dim=-1)[0])`
- **Weighting**: Combine this with the `recon_loss` (Huber).
- **Expected Impact**: Forces the model to respect the "extreme physics" of the peak demand, preventing the over-smoothing that caps performance.

## 🟢 3. Seasonal Intelligence
**Problem**: The model currently relies on the temperature float to infer seasonality.

**Technical Spec**:
- **Condition Expansion**: Increase `COND_DIM` from 6 to 8.
- **New Flags**:
    - `is_summer`: Boolean (Months 6, 7, 8).
    - `is_winter`: Boolean (Months 12, 1, 2).
- **Expected Impact**: Provides a discrete contextual clue to help the decoder separate high-variance seasonal baseline loads (Heating vs. AC).

---

## 🔴 4. "Red Flag" Constraint: No Capacity Expansion
**Restriction**: Do **NOT** increase `DECODER_HIDDEN` or add more TCN layers.
**Rationale**: Expanding model capacity (MoE or Deep Transformers) with only 5,000 samples will trigger **Posterior Collapse** and "Model Obesity." We stay lean at ~4M parameters and use better loss physics (Peak Loss) and better feature anchoring (PE) to solve the accuracy gap instead.

---

## Next Steps for Human implementation
1. Update `CFG.EPOCHS = 300`.
2. Update `CFG.COND_DIM = 8`.
3. Add `PositionalEncoding` logic to model definitions.
4. Add `peak_loss` calculation to the `train_evolvai` loop.
