import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class CFG:
    NUM_NODES = 32
    SEQ_LEN = 24
    NUM_WEATHER_FEAT = 4
    NUM_FEATURES = 36
    COND_DIM = 8
    TCN_CHANNELS = [128, 256, 256, 256]
    KERNEL_SIZE = 3
    DROPOUT = 0.0
    LATENT_DIM = 128
    DECODER_HIDDEN = 512

with open('Latest_Training.txt', 'r') as f:
    import json
    nb = json.loads(f.read())
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            src = "".join(cell['source'])
            if 'class GenerativeCounterfactualVAE' in src:
                exec(src, globals())

m = GenerativeCounterfactualVAE()
optimizer = torch.optim.Adam(m.parameters(), lr=2e-4)
x = torch.randn(8, 36, 24)
cond = torch.randn(8, 8)

for i in range(101):
    optimizer.zero_grad()
    recon, mu, lv = m(x, cond)
    loss = F.huber_loss(recon, x)
    loss.backward()
    optimizer.step()
    if i % 20 == 0:
        print(f"Iter {i}, Loss: {loss.item()}")
