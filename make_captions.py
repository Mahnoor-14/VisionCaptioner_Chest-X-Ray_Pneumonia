# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 14:36:53 2025

@author: Maviya
"""

import os
import pandas as pd

# Example: chest_xray dataset → labels as captions
root = r"D:\use_case\data\chest_xray\train"

rows = []
for label in ["NORMAL", "PNEUMONIA"]:
    folder = os.path.join(root, label)
    for f in os.listdir(folder):
        rows.append([os.path.join(folder, f),
                     "no finding" if label=="NORMAL" else "pneumonia detected"])

df = pd.DataFrame(rows, columns=["image_path", "caption"])
df.to_csv(r"D:\use_case\data/captions.csv", index=False)
print("Saved data/captions.csv with", len(df), "samples")
