import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datasets import load_dataset, concatenate_datasets
import pandas as pd
import torch.nn.functional as F

# Configurazione Hardware
# device = "cuda" if torch.cuda.is_available() else "cpu"
# #print(f"Utilizzo dispositivo: {device}")
#
# # load model
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
#
# #print(model)
# #print(processor)
#
# # load dataset
# ds = load_dataset("jxie/flickr8k")
#
# # Uniamo tutto in un unico dataset "globale" per avere più immagini nel database
# full_dataset = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

#print(full_dataset)

# Semplice e diretto: Pandas converte il dizionario di liste in righe e colonne
#df_stats = pd.DataFrame(full_dataset[1:10])

#print(df_stats)

BATCH_SIZE = 32

# non usiamo le caption? Bisogna riflettere

# dato che stiamo usando hugging face approcciamo l'uso delle sue funzioni (altrimenti avrei usato il dataloader di pytorch)

def compute_features(batch, model, processor, device):
    images = [img.convert("RGB") for img in batch["image"]]
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output = model.get_image_features(pixel_values=inputs["pixel_values"])
        feats = output.pooler_output           # [batch, 512]
        #print(feats.shape)
        feats = F.normalize(feats, p=2, dim=1)
    return {"embeddings": feats.cpu().numpy()}

def build_embeddings(model, processor, full_dataset, device):
    # .map() non accetta parametri extra direttamente, usiamo lambda
    updated_dataset = full_dataset.map(
        lambda batch: compute_features(batch, model, processor, device),
        batched=True,
        batch_size=32,
        desc="Computing embeddings"
    )
    print(updated_dataset["embeddings"])

    embeddings = np.array(updated_dataset["embeddings"])  # [8000, 512]
    np.save("embeddings.npy", embeddings)
    print(f" Salvati {embeddings.shape[0]} embedding di dimensione {embeddings.shape[1]}")