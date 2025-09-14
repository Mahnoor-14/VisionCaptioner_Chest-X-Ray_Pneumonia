#  VisionCaptioner - Chest X-Ray Pneumonia Captioning (CNN + Transformer)

## 📌 Project Overview
**VisionCaptioner** is an AI-based image captioning system that combines **ResNet-50** (CNN encoder) with a **Transformer decoder** to generate human-like captions for images.  
This repository focuses on **medical imaging** — particularly **Chest X-Rays** — to automatically describe whether an image is **Normal** or shows signs of **Pneumonia**.  

The project supports training, inference, and a **Streamlit frontend** for interactive testing.

---

## ✨ Features
- 📂 Upload an X-ray image via web interface  
- 🩻 Predict captions such as **“no finding”** or **“pneumonia detected”**  
- 🔎 Transformer decoder with attention mechanism for interpretability  
- 🖥️ Lightweight Streamlit web app for clinicians/researchers  
- 🎯 Extendable to other datasets (e.g., COCO, satellite imagery)  

---

## 🛠️ Tech Stack
- **Language**: Python 3.9+  
- **Frameworks**: PyTorch, Torchvision  
- **Frontend**: Streamlit  
- **Data Handling**: Pandas, Pillow, Numpy  
- **Visualization**: Matplotlib (optional)  

---

## Install Dependencies
pip install -r requirements.txt

## Frontend
Launch the interactive frontend:

streamlit run frontend_captions.py
Browser opens at: 👉 http://localhost:8501

## 🚀 Installation
Clone the repository:

git clone https://github.com/YOUR-USERNAME/VisionCaptioner_Chest-X-Ray_Pneumonia.git
cd VisionCaptioner_Chest-X-Ray_Pneumonia

## Pretrained Model


👉 Download best_model.pth + vocab.pkl (https://drive.google.com/drive/folders/1fbGgpFKKZG2zWdeFGvGDtBCtEyvsMzlw?usp=sharing)

After downloading, place them into:
- checkpoints/best_model.pth
- checkpoints/vocab.pkl
