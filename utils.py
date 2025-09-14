# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:41:23 2025

@author: Maviya
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.idx2word = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.idx = 4

    def add_sentence(self, sentence):
        for word in sentence.lower().split():
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def numericalize(self, sentence):
        return [self.word2idx.get(word, 3) for word in sentence.lower().split()]

class CaptionDataset(Dataset):
    def __init__(self, csv_file, vocab, transform=None):
        self.df = pd.read_csv(csv_file)
        self.vocab = vocab
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        for cap in self.df["caption"]:
            vocab.add_sentence(cap)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        caption = [1] + self.vocab.numericalize(row["caption"]) + [2]  # <SOS> ... <EOS>
        return image, torch.tensor(caption)
