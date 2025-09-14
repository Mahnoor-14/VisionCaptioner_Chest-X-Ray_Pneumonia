# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:40:41 2025

@author: Maviya
"""

import torch
import torch.nn as nn

class DecoderTransformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = nn.Sequential(
            nn.Dropout(dropout)
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)  # (T, N, E)
        embeddings = self.pos_encoder(embeddings)
        tgt_mask = nn.Transformer().generate_square_subsequent_mask(embeddings.size(0)).to(embeddings.device)
        out = self.transformer_decoder(embeddings, features.unsqueeze(0), tgt_mask=tgt_mask)
        return self.fc(out)
