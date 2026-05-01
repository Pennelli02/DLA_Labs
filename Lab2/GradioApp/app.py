import gradio as gr
from retrieve_logic import search

def retrieve(prompt, n_img):
    if not prompt.strip():
        return []
    return search(prompt, n_img)

with gr.Blocks() as demo:
    gr.Markdown("# Fancy Text-to-Image Retrieval")
    gr.Markdown("Cerca immagini dal dataset Flickr8k usando una descrizione testuale")
    with gr.Row(equal_height=True):
        prompt = gr.Textbox(label="Prompt", scale=9)
        retr_btn = gr.Button("Retrieve", scale=1)
    with gr.Row():
        n_img = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Numero immagini")
    output = gr.Gallery(
        label="Risultati",
        columns=5,
        rows=2,
        height=500,
        object_fit="cover"  # tutte le immagini stessa dimensione
    )

    retr_btn.click(fn=retrieve, inputs=[prompt, n_img], outputs=[output])
    prompt.submit(fn=retrieve, inputs=[prompt, n_img], outputs=[output])

demo.launch(theme=gr.themes.Soft())