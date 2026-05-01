import os
from builder_index import build_embeddings
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, concatenate_datasets

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# path assoluto della cartella dove si trova retrieve_logic.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings.npy")

#embeddings = np.load("embeddings.npy")  # [8000, 512]

ds = load_dataset("jxie/flickr8k")
full_dataset = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

def load_index():
    if os.path.exists(EMBEDDINGS_PATH):
        print("embeddings.npy trovato, lo carico...")
        return np.load(EMBEDDINGS_PATH)
    else:
        return build_embeddings(model=model, processor=processor, full_dataset=full_dataset, device=device)

# viene fatto una volta sola
embeddings = load_index()

def get_text_embedding(prompt):
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_feats = model.get_text_features(**inputs)
        text_feats = text_feats.pooler_output
        text_feats = F.normalize(text_feats, p=2, dim=1)
    return text_feats.cpu().numpy()  # [1, 512]

def search(prompt, topk):
    text_emb = get_text_embedding(prompt)
    scores = (embeddings @ text_emb.T)[:, 0]
    top_indices = scores.argsort()[::-1][:int(topk)]
    return [full_dataset[int(i)]["image"] for i in top_indices]