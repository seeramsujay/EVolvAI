import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

with open('Latest_Training.txt', 'r') as f:
    import json
    nb = json.loads(f.read())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'class CFG' in src or 'class GenerativeCounterfactualVAE' in src:
                exec(src, globals())

# Test with dropout = 0.15
CFG.KLD_MAX = 0.0
CFG.DROPOUT = 0.15
m = GenerativeCounterfactualVAE()
optimizer = torch.optim.Adam(m.parameters(), lr=2e-4)

# Generate a fixed batch to memorize
x = torch.randn(64, 36, 24)
cond = torch.randn(64, 8)

for i in range(101):
    m.train()
    optimizer.zero_grad()
    recon, mu, lv = m(x, cond)
    loss = F.huber_loss(recon, x)
    loss.backward()
    optimizer.step()
    if i == 100:
        print(f"Dropout 0.15 Final Loss: {loss.item()}")

# Test with dropout = 0.0
CFG.DROPOUT = 0.0
m2 = GenerativeCounterfactualVAE()
optimizer2 = torch.optim.Adam(m2.parameters(), lr=2e-4)

for i in range(101):
    m2.train()
    optimizer2.zero_grad()
    recon, mu, lv = m2(x, cond)
    loss = F.huber_loss(recon, x)
    loss.backward()
    optimizer2.step()
    if i == 100:
        print(f"Dropout 0.00 Final Loss: {loss.item()}")
