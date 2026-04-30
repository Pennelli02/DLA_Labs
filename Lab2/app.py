import gradio as gr
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from datasets import load_dataset, concatenate_datasets

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# carica embeddings pre-calcolati
embeddings = np.load("embeddings.npy")  # [8000, 512]

# carica dataset per recuperare le immagini dato l'indice
ds = load_dataset("jxie/flickr8k")
full_dataset = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

# L'idea è che per ogni immagine e prompt otteniamo e salviamo il similarity score dell'immagini del nostro database e li salvo.
# in fase di retrieval passo ogni immagine nel prompt inserito ottengo le similarity e poi confronto quale tra loro sono le più simili
# Non so se sia la soluzione più efficiente.
# TODO creare funzione che usa questi image_embedding e attraverso il prompt inserito ottiene il text_embedding che verrà confrontato
# TODO add ulteriori features per l'app per estetica (tipo es di prompt, num di immagini ecc...)
def retrieve(prompt):
    return "Hello, " + prompt + "!"

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt")
    output = gr.Image(label="Similar images")
    retr_btn = gr.Button("retrieve")

    retr_btn.click(fn=retrieve, inputs=[prompt], outputs=[output])
    prompt.submit(fn=retrieve, inputs=[prompt], outputs=[output])


demo.launch()

