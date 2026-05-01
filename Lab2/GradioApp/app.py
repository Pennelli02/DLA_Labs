import gradio as gr
from retrieve_logic import search

def retrieve(prompt, n_img):
    return search(prompt, n_img)

with gr.Blocks() as demo:
    gr.Markdown("# Fancy Text-to-Image Retrieval")
    with gr.Row(equal_height=True):
        prompt = gr.Textbox(label="Prompt", scale=8)
        retr_btn = gr.Button("Retrieve", scale=1)
    with gr.Row():
        n_img = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Numero immagini")
    output = gr.Gallery(label="Similar images", height=500)

    retr_btn.click(fn=retrieve, inputs=[prompt, n_img], outputs=[output])
    prompt.submit(fn=retrieve, inputs=[prompt, n_img], outputs=[output])

demo.launch()