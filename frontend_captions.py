import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.caption_model import ImageCaptioningModel
import pickle
import os

st.title("🩻 VisionCaptioner - X-ray Caption Generator")

device = "cuda" if torch.cuda.is_available() else "cpu"

# 🔹 Load saved vocabulary
with open("checkpoints/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# 🔹 Load model with correct vocab size
model = ImageCaptioningModel(256, len(vocab.word2idx)).to(device)

if os.path.exists("checkpoints/best_model.pth"):
    model.load_state_dict(torch.load("checkpoints/best_model.pth", map_location=device))
    model.eval()
else:
    st.error("❌ best_model.pth not found in checkpoints/")

def generate_caption(img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model.encoder(img)
        caption_idx = [vocab.word2idx["<SOS>"]]
        for _ in range(20):
            inputs = torch.tensor(caption_idx).unsqueeze(1).to(device)
            output = model.decoder(feature, inputs)
            next_word = output[-1].argmax().item()
            caption_idx.append(next_word)
            if next_word == vocab.word2idx["<EOS>"]:
                break
    return " ".join([vocab.idx2word[i] for i in caption_idx if i not in [0,1,2,3]])

uploaded = st.file_uploader("📂 Upload an X-ray image", type=["jpg","png","jpeg"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)
    st.write("### 📝 Generated Caption:")
    st.success(generate_caption(img))
