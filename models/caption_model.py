# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:41:00 2025

@author: Maviya
"""

import torch.nn as nn
from models.encoder import EncoderCNN
from models.decoder import DecoderTransformer

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderTransformer(embed_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
