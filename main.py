# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:41:49 2025

@author: Maviya
"""

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils import CaptionDataset, Vocabulary
from models.caption_model import ImageCaptioningModel

csv = "data/captions.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset + Vocab
vocab = Vocabulary()
dataset = CaptionDataset(csv, vocab)
loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=lambda x: zip(*x))
# 🔹 Save vocab so we can reuse in frontend/inference
import pickle, os
os.makedirs("checkpoints", exist_ok=True)

with open("checkpoints/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("✅ Saved vocab.pkl with size:", len(vocab.word2idx))
#%%
# Model
embed_size = 256
model = ImageCaptioningModel(embed_size, len(vocab.word2idx)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(5):
    for imgs, caps in loader:
        imgs = torch.stack(list(imgs)).to(device)
        caps = torch.nn.utils.rnn.pad_sequence(list(caps), batch_first=False).to(device)

        outputs = model(imgs, caps[:-1])
        loss = criterion(outputs.view(-1, outputs.size(-1)), caps[1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/best_model.pth")
