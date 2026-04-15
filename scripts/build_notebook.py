"""
build_notebook.py  — regenerates EVolvAI_Training.ipynb from train.py logic
Run from repo root: python3 build_notebook.py
"""
import json, os

# Since this is in scripts/, root is one level up
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def code(src): 
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": source}

def md(src):
    lines = src.split("\n")
    source = [l + "\n" for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    return {"cell_type": "markdown", "metadata": {}, "source": source}

CELLS = []

# ── Title ─────────────────────────────────────────────────────────────────────
CELLS.append(md("""\
# 🔋 EVolvAI — Physics-Informed TCN-VAE
## Self-contained Colab training notebook  ·  mirrors `train.py` exactly

Pipeline:
1. Clone repo → import all modules from `generative_core/`
2. Build dataset — real parquet if present, synthetic fallback otherwise
3. Train with **KLD annealing** (β: 0→1) + **physics annealing** (λ: 0→full)
4. Save checkpoint + generate counterfactual scenarios
5. Inline quality check — then `python3 tester.py` for the full report

> 💡 Edit **only Cell 3 (Config)**. Run all: `Ctrl+F9`.
"""))

# ── 0: install ─────────────────────────────────────────────────────────────────
CELLS.append(md("## 0 · Install & Imports"))
CELLS.append(code("""\
!pip install -q torch numpy pandas pyarrow scipy matplotlib requests
import sys, os, time, datetime
import numpy as np
import torch
print("Python :", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("CUDA   :", torch.cuda.is_available())
"""))

# ── 1: mount drive ─────────────────────────────────────────────────────────────
CELLS.append(md("## 1 · Mount Drive *(optional but recommended)*"))
CELLS.append(code("""\
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_DIR = "/content/drive/MyDrive/EVolvAI_output"
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print("Drive mounted ✓  backups →", DRIVE_DIR)
except Exception as e:
    DRIVE_DIR = None
    print("Drive not mounted:", e)
"""))

# ── 2: clone ───────────────────────────────────────────────────────────────────
CELLS.append(md("## 2 · Clone Repo"))
CELLS.append(code("""\
REPO_URL = "https://github.com/seeramsujay/EVolvAI.git"
REPO_DIR = "/content/EVolvAI"

if not os.path.isdir(REPO_DIR):
    !git clone --depth 1 {REPO_URL} {REPO_DIR}
else:
    !git -C {REPO_DIR} pull

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

os.chdir(REPO_DIR)
print("Working directory:", os.getcwd())
"""))

# ── 3: config ──────────────────────────────────────────────────────────────────
CELLS.append(md("## 3 · Config — *edit only here*"))
CELLS.append(code("""\
# ── TRAINING ─────────────────────────────────────────────────────────────────
EPOCHS         = 500       # 500 ≈ 2-3 hrs CPU | ~30 min T4 GPU
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-3
GRAD_CLIP_NORM = 1.0
LOG_EVERY      = 25        # print progress every N epochs
OUTPUT_DIR     = os.path.join(REPO_DIR, "output")

# ── KLD ANNEALING ─────────────────────────────────────────────────────────────
# β ramps linearly from 0 → KLD_MAX over KLD_ANNEAL_EPOCHS epochs.
# Prevents posterior collapse: decoder learns reconstruction *before*
# the prior enforces structure.
KLD_MAX          = 1.0
KLD_ANNEAL_EPOCHS = 100

# ── PHYSICS ANNEALING ─────────────────────────────────────────────────────────
# λ ramps from 0 → full over PHYS_ANNEAL_EPOCHS.
# Physics lambdas in config.py are set to 0.0 (isolation phase).
# This annealing re-enables them gradually once the VAE has learned to reconstruct.
PHYS_ANNEAL_EPOCHS = 150

# ── LR SCHEDULE ──────────────────────────────────────────────────────────────
LR_STEP  = 150    # halve LR every N epochs
LR_GAMMA = 0.5

print(f"Epochs={EPOCHS}  Batch={BATCH_SIZE}  LR={LEARNING_RATE}")
print(f"KLD  anneal: β 0→{KLD_MAX}  over {KLD_ANNEAL_EPOCHS} epochs")
print(f"Phys anneal: λ 0→1.0 over {PHYS_ANNEAL_EPOCHS} epochs")
"""))

# ── 4: import modules ──────────────────────────────────────────────────────────
CELLS.append(md("## 4 · Import Repo Modules"))
CELLS.append(code("""\
from generative_core import config as CFG
from generative_core.data_loader import get_dataloader
from generative_core.models import GenerativeCounterfactualVAE, vae_loss_function
from generative_core.physics_loss import LinDistFlowLoss
import torch.optim as optim

# Push notebook config into CFG
CFG.EPOCHS        = EPOCHS
CFG.BATCH_SIZE    = BATCH_SIZE
CFG.LEARNING_RATE = LEARNING_RATE

print("Imports OK ✓")
print(f"  LATENT_DIM    = {CFG.LATENT_DIM}")
print(f"  TCN_CHANNELS  = {CFG.TCN_CHANNELS}")
print(f"  NUM_FEATURES  = {CFG.NUM_FEATURES}")
print(f"  COND_DIM      = {CFG.COND_DIM}")
print(f"  DECODER_HIDDEN= {CFG.DECODER_HIDDEN}")
parquet_ok = os.path.isfile(CFG.DATA_PATH)
print(f"  Data source   : {'real parquet ✅' if parquet_ok else 'synthetic fallback ⚠'}")
"""))

# ── 5: build dataset ───────────────────────────────────────────────────────────
CELLS.append(md("""\
## 5 · Build Dataset

`get_dataloader()` automatically:
- Reads `data/processed/train_data.parquet` if present (real ACN-Data)
- Falls back to the synthetic Lochan generator otherwise

Each batch → `(x, cond)` where `cond` is a **dynamic, per-date condition vector**
computed from the actual date in the parquet (weekend flag, solar from DOY, traffic pattern).
"""))
CELLS.append(code("""\
torch.manual_seed(42)
np.random.seed(42)

loader = get_dataloader(batch_size=BATCH_SIZE)
print(f"Batches/epoch : {len(loader)}")
"""))

# ── 6: build model ─────────────────────────────────────────────────────────────
CELLS.append(md("## 6 · Build Model"))
CELLS.append(code("""\
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model          = GenerativeCounterfactualVAE().to(device)
optimizer      = optim.Adam(model.parameters(), lr=LEARNING_RATE)
physics_engine = LinDistFlowLoss(device)
scheduler      = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters : {total_params:,}")
print(f"TCN        : {CFG.TCN_CHANNELS}")
print(f"Latent dim : {CFG.LATENT_DIM}")
"""))

# ── 7: training loop ──────────────────────────────────────────────────────────
CELLS.append(md("""\
## 7 · Training Loop

Two simultaneous annealing schedules:
- **β (KLD)**: 0→1 over first 100 epochs — decoder learns freely first
- **λ (physics)**: 0→full over first 150 epochs — grid constraints introduced gently

Since `LAMBDA_VOLT/THERMAL/XFMR` are set to **0.0** in `config.py` (isolation phase),
the physics annealing here acts as the only physics pressure — and it starts at zero.
"""))
CELLS.append(code("""\
# Store base lambdas from config (all 0.0 during isolation, re-enable later)
_lv = CFG.LAMBDA_VOLT
_lt = CFG.LAMBDA_THERMAL
_lx = CFG.LAMBDA_XFMR

history = []
t_start = time.time()

model.train()
for epoch in range(1, EPOCHS + 1):

    # ── KLD annealing: β 0 → KLD_MAX ─────────────────────────────────────
    beta = min(1.0, epoch / KLD_ANNEAL_EPOCHS) * KLD_MAX

    # ── Physics annealing: λ 0 → full ────────────────────────────────────
    lam = min(1.0, epoch / PHYS_ANNEAL_EPOCHS)
    CFG.LAMBDA_VOLT    = _lv * lam
    CFG.LAMBDA_THERMAL = _lt * lam
    CFG.LAMBDA_XFMR    = _lx * lam

    epoch_loss = epoch_phys = 0.0
    n_batches  = 0

    for x, cond in loader:
        # x    : [B, NUM_FEATURES, SEQ_LEN] — channel-first (from EVDemandDataset)
        # cond : [B, COND_DIM]              — dynamic per-date condition
        x    = x.to(device)
        cond = cond.to(device)

        optimizer.zero_grad()
        recon, mu, logvar = model(x, cond)

        ev_demand = recon[:, :CFG.NUM_NODES, :].permute(0, 2, 1)  # [B, 24, 32]
        pen_v, pen_therm, pen_xfmr = physics_engine(ev_demand)
        phys = (CFG.LAMBDA_VOLT    * pen_v
              + CFG.LAMBDA_THERMAL * pen_therm
              + CFG.LAMBDA_XFMR   * pen_xfmr)

        loss = vae_loss_function(recon, x, mu, logvar, phys,
                                 current_kld_weight=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        epoch_loss += loss.item()
        epoch_phys += phys.item()
        n_batches  += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)
    avg_phys = epoch_phys / max(n_batches, 1)
    history.append(avg_loss)

    if epoch % LOG_EVERY == 0 or epoch == 1:
        elapsed = (time.time() - t_start) / 60
        eta     = (elapsed / epoch) * (EPOCHS - epoch)
        lr_now  = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:>4}/{EPOCHS}  "
              f"loss={avg_loss:.5f}  phys={avg_phys:.5f}  "
              f"β={beta:.2f}  λ={lam:.2f}  lr={lr_now:.1e}  "
              f"elapsed={elapsed:.1f}min  ETA={eta:.1f}min")

        # Mid-run Drive backup every 100 epochs
        if DRIVE_DIR and epoch % 100 == 0:
            import shutil
            _bk = os.path.join(DRIVE_DIR, f"gcvae_ep{epoch}.pt")
            torch.save(model.state_dict(), _bk)
            print(f"    Drive backup → {_bk}")

total_min = (time.time() - t_start) / 60
print(f"\\nTraining done in {total_min:.1f} min ✓")
"""))

# ── 8: save + generate ────────────────────────────────────────────────────────
CELLS.append(md("## 8 · Save Checkpoint + Generate Counterfactuals"))
CELLS.append(code("""\
import shutil

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Checkpoint
ckpt = os.path.join(OUTPUT_DIR, "gcvae_model.pt")
torch.save(model.state_dict(), ckpt)
print(f"Checkpoint → {ckpt}")
if DRIVE_DIR:
    shutil.copy(ckpt, os.path.join(DRIVE_DIR, "gcvae_model.pt"))

# Loss CSV
csv_path = os.path.join(OUTPUT_DIR, "training_loss.csv")
with open(csv_path, "w") as f:
    f.write("epoch,avg_loss\\n")
    for i, v in enumerate(history, 1):
        f.write(f"{i},{v:.6f}\\n")
print(f"Loss CSV  → {csv_path}")

# Counterfactual scenarios
print("\\nGenerating counterfactuals…")
model.eval()
for name, spec in CFG.SCENARIOS.items():
    with torch.no_grad():
        z    = torch.randn(1, CFG.LATENT_DIM, device=device)
        cond = torch.tensor([spec["condition"]], dtype=torch.float32, device=device)
        out  = model.decode(z, cond)
        demand = out[:, :CFG.NUM_NODES, :].squeeze(0).permute(1, 0).cpu().numpy()

    path = os.path.join(OUTPUT_DIR, f"{name}.npy")
    np.save(path, demand)
    zpct = (demand == 0).mean() * 100
    flag = "✅" if zpct < 30 else "⚠️ "
    print(f"  {flag} [{name:30s}] range=[{demand.min():.3f},{demand.max():.3f}]  zeros={zpct:.1f}%")
"""))

# ── 9: loss plot ──────────────────────────────────────────────────────────────
CELLS.append(md("## 9 · Loss Curve"))
CELLS.append(code("""\
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(history, lw=1.5, color="#4f86c6")
plt.xlabel("Epoch"); plt.ylabel("Avg Loss")
plt.title("EVolvAI Training Loss — Physics-Informed TCN-VAE")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
plt.show()
print("Saved training_loss.png")
"""))

# ── 10: inline quality check ──────────────────────────────────────────────────
CELLS.append(md("## 10 · Inline Quality Check"))
CELLS.append(code("""\
scenarios = [
    "extreme_winter_storm", "summer_peak",
    "full_electrification", "extreme_winter_v2", "rush_hour_gridlock",
]

print("=" * 62)
print("  Post-training Quality Check")
print("=" * 62)
zero_pcts = []
for name in scenarios:
    p = os.path.join(OUTPUT_DIR, f"{name}.npy")
    if not os.path.exists(p):
        continue
    a = np.load(p)
    z = (a == 0).mean() * 100
    zero_pcts.append(z)
    flag = "✅" if z < 30 else "🟡" if z < 50 else "🔴"
    print(f"  {flag} {name:32s}  zeros={z:.1f}%  range=[{a.min():.3f},{a.max():.3f}]")

if zero_pcts:
    avg_z = np.mean(zero_pcts)
    print(f"\\n  Avg zeros : {avg_z:.1f}%")
    if avg_z < 20:
        print("  🟢  HEALTHY — ready for full 1000-epoch GPU run or handoff")
    elif avg_z < 40:
        print("  🟡  PARTIAL — run more epochs")
    else:
        print("  🔴  COLLAPSE — both annealings are ON; needs more epochs")
print("=" * 62)
"""))

# ── 11: download ──────────────────────────────────────────────────────────────
CELLS.append(md("## 11 · Download Outputs"))
CELLS.append(code("""\
zip_path = "/content/EVolvAI_output"
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
print(f"Archive: {zip_path}.zip")
try:
    from google.colab import files
    files.download(f"{zip_path}.zip")
except ImportError:
    print("Not in Colab — outputs at:", OUTPUT_DIR)
"""))

# ─── Assemble ─────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "toc_visible": True},
    },
    "cells": CELLS,
}

out = os.path.join(ROOT, "EVolvAI_Training.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"✅  Notebook written → {out}  ({len(CELLS)} cells)")
