"""
tester.py — EVolvAI Output Quality Checker
==========================================
Run from repo root any time after a Colab training run:

    python3 tester.py

Checks:
  - Shape & range of every scenario .npy
  - Zero-fraction (posterior collapse detector)
  - Temporal + spatial variation
  - Scenario differentiation (condition vector health)
  - Model checkpoint metadata
  - Quality verdict: 🟢 / 🟡 / 🔴
"""

import os
import json
import datetime
import numpy as np

OUTPUT = os.path.join(os.path.dirname(__file__), "output")

# ─── Scenarios to check ────────────────────────────────────────────────────────
EXPECTED_SCENARIOS = [
    "extreme_winter_storm",
    "summer_peak",
    "full_electrification",
    "extreme_winter_v2",
    "rush_hour_gridlock",
]

EXPECTED_SHAPE = (24, 32)   # [hours, IEEE-33-bus load nodes]

W = 64   # column width for formatting

def banner(text):
    print("\n" + "─" * W)
    print(f"  {text}")
    print("─" * W)

def run():
    print("=" * W)
    print("  EVolvAI — Output Quality Report")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print("=" * W)

    # ── 1. Load scenario arrays ───────────────────────────────────────────────
    banner("Scenario Arrays")

    arrays   = {}
    missing  = []
    verdicts = []   # per-scenario pass/fail/warn

    for name in EXPECTED_SCENARIOS:
        path = os.path.join(OUTPUT, f"{name}.npy")
        if not os.path.exists(path):
            print(f"  ❌  {name:35s} MISSING")
            missing.append(name)
            continue

        a        = np.load(path)          # expected [24, 32]
        arrays[name] = a

        hourly   = a.mean(axis=1)         # [24]  — mean kW across nodes per hour
        nodal    = a.mean(axis=0)         # [32]  — mean kW per node across day
        zero_pct = (a == 0).mean() * 100
        temp_std = hourly.std()
        node_std = nodal.std()
        peak_h   = hourly.argmax()

        # Shape check
        shape_ok = (a.shape == EXPECTED_SHAPE)

        # Health flags
        collapsed   = zero_pct > 50
        poor_temp   = temp_std < 0.01
        poor_node   = node_std < 0.001

        if not shape_ok:
            flag = "❌"
        elif collapsed or poor_temp:
            flag = "⚠️ "
        else:
            flag = "✅"

        verdicts.append(flag)

        print(f"\n  {flag} [{name}]")
        print(f"       shape     : {a.shape}  {'✓' if shape_ok else '✗ WRONG SHAPE'}")
        print(f"       range     : [{a.min():.4f},  {a.max():.4f}]")
        print(f"       mean      : {a.mean():.4f}")
        print(f"       zeros     : {zero_pct:5.1f}%  "
              f"{'🔴 posterior collapse' if zero_pct > 50 else ('🟡 partially collapsed' if zero_pct > 30 else '🟢 healthy')}")
        print(f"       temp_std  : {temp_std:.4f}  (temporal variation — want > 0.05)")
        print(f"       node_std  : {node_std:.4f}  (spatial spread — want > 0.001)")
        print(f"       peak_hour : {peak_h}:00  {'✓ realistic (16-21h)' if 16 <= peak_h <= 21 else '⚠ unexpected peak hour'}")


    # ── 2. Scenario differentiation ───────────────────────────────────────────
    banner("Scenario Differentiation  (condition vector health)")

    if len(arrays) >= 2:
        names = list(arrays.keys())
        means = [arrays[n].mean() for n in names]
        diffs = [abs(means[i] - means[j])
                 for i in range(len(means)) for j in range(i+1, len(means))]
        max_d = max(diffs)
        min_d = min(diffs)

        print(f"  Scenarios loaded   : {len(arrays)}/{len(EXPECTED_SCENARIOS)}")
        print(f"  Mean output range  : {min(means):.4f} – {max(means):.4f}")
        print(f"  Max Δ between any two scenarios : {max_d:.4f}")
        print(f"  Min Δ between any two scenarios : {min_d:.4f}")

        if max_d > 0.05:
            print("  ✅  Condition vector is working — scenarios are differentiated")
        elif max_d > 0.01:
            print("  🟡  Scenarios slightly different — more epochs may help")
        else:
            print("  🔴  Scenarios look IDENTICAL — condition vector has no effect")
    else:
        print("  Not enough scenarios loaded to compare.")


    # ── 3. Checkpoint metadata ────────────────────────────────────────────────
    banner("Model Checkpoint")

    ckpt_path = os.path.join(OUTPUT, "gcvae_model.pt")
    if os.path.exists(ckpt_path):
        size_kb  = os.path.getsize(ckpt_path) / 1024
        mtime    = datetime.datetime.fromtimestamp(os.path.getmtime(ckpt_path))
        print(f"  Path      : {ckpt_path}")
        print(f"  Size      : {size_kb:.1f} kB")
        print(f"  Saved at  : {mtime.strftime('%Y-%m-%d %H:%M:%S')}")

        # Try to inspect state dict keys
        try:
            import torch
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            print(f"  Keys      : {len(sd)}  (parameter tensors)")
            total_params = sum(v.numel() for v in sd.values())
            print(f"  Params    : {total_params:,}")
        except Exception as e:
            print(f"  (Could not inspect state dict: {e})")
    else:
        print("  ❌  gcvae_model.pt not found in output/")


    # ── 4. Overall verdict ────────────────────────────────────────────────────
    banner("Overall Verdict")

    if not arrays:
        print("  🔴  No scenario arrays found — nothing to evaluate.")
        return

    avg_zeros = np.mean([(arrays[n] == 0).mean() * 100 for n in arrays])
    n_missing = len(missing)
    n_bad_shape = sum(1 for n in arrays if arrays[n].shape != EXPECTED_SHAPE)

    print(f"  Scenarios found    : {len(arrays)}/{len(EXPECTED_SCENARIOS)}")
    print(f"  Missing            : {n_missing}")
    print(f"  Wrong shape        : {n_bad_shape}")
    print(f"  Avg zero-fraction  : {avg_zeros:.1f}%")

    if n_missing > 0 or n_bad_shape > 0:
        print("\n  🔴  FIX ERRORS FIRST before training further.")
    elif avg_zeros < 20:
        print("\n  🟢  Model looks healthy!")
        print("      → Ready for a full 1000-epoch GPU run.")
        print("      → Outputs are safe to hand off to the risk engine & dashboard.")
    elif avg_zeros < 40:
        print("\n  🟡  Partially converged.")
        print("      → Run 500+ epochs with KLD annealing enabled.")
        print("      → Do NOT hand off to risk engine yet.")
    else:
        print("\n  🔴  Posterior collapse detected.")
        print("      → Ensure KLD annealing is ON (β starts at 0).")
        print("      → Reduce LAMBDA_VOLT / LAMBDA_THERMAL / LAMBDA_XFMR by 0.5x.")
        print("      → Re-run training from scratch.")

    print("\n" + "=" * W)


if __name__ == "__main__":
    run()
