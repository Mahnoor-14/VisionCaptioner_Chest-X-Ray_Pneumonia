# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:42:31 2025

@author: Maviya
"""

import torch
from PIL import Image
from torchvision import transforms
from utils import Vocabulary
from models.caption_model import ImageCaptioningModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load vocab + model
vocab = Vocabulary()
model = ImageCaptioningModel(256, len(vocab.word2idx)).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
model.eval()

def generate_caption(image_path):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model.encoder(img)
        caption_idx = [1]  # <SOS>
        for _ in range(20):
            inputs = torch.tensor(caption_idx).unsqueeze(1).to(device)
            output = model.decoder(feature, inputs)
            next_word = output[-1].argmax().item()
            caption_idx.append(next_word)
            if next_word == 2: break  # <EOS>

    return " ".join([vocab.idx2word[i] for i in caption_idx if i>3])

print(generate_caption("data/chest_xray/test/NORMAL/img.jpeg"))
