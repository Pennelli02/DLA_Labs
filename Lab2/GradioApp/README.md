# GradioApp — Text-to-Image Retrieval (Exercise 3.3)
 
A simple **text-to-image retrieval** web app built with **CLIP** and **Gradio**. It lets a user type a natural-language prompt (e.g. *"a dog playing in the snow"*) and returns the most visually-relevant images from the [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) dataset.
 
---
 
## How it works
 
The app is built around a simple **index once, query many times** pattern:
 
1. **Indexing (offline, done once)** — every image in Flickr8k is passed through CLIP's image encoder to obtain a 512-dimensional, L2-normalized embedding. All embeddings are stacked into a single matrix and cached to disk as `embeddings.npy`, so this expensive step only needs to run once.
2. **Querying (online, per request)** — when the user submits a text prompt, it is encoded with CLIP's text encoder into the same 512-dimensional embedding space. **Cosine similarity** (a simple dot product, since all vectors are L2-normalized) is computed between the prompt embedding and every cached image embedding, and the top-K most similar images are returned and displayed in the UI.
This works because CLIP is trained to place matching images and text captions close together in the same embedding space, so "similar direction" ≈ "similar meaning."
 
---
 
## Files
 
### `app.py`
The **Gradio user interface**. Defines a simple `gr.Blocks` layout with:
- a text box for the prompt,
- a slider to choose how many images to retrieve (1–50, default 10),
- a "Retrieve" button (the search also triggers on pressing Enter in the text box),
- a `Gallery` component (5 columns × 2 rows) to display the returned images.
It contains no retrieval logic itself — it simply calls `search()` from `retrieve_logic.py` and forwards the results to the gallery. Empty prompts are short-circuited to return no results.
 
### `builder_index.py`
Responsible for **building the embedding index** from scratch. Given a CLIP model, processor, dataset, and device, it:
1. Loads the Flickr8k dataset (train + validation + test splits, concatenated into a single pool of images via `concatenate_datasets`, to have as many indexed images as possible).
2. Processes images in batches (`BATCH_SIZE = 32`) through `dataset.map(...)`, using `compute_features()` to run each batch through CLIP's image encoder (`model.get_image_features`) and L2-normalize the resulting features.
3. Stacks all batch embeddings into a single NumPy array of shape `[8000, 512]` and saves it to `embeddings.npy`.
This module is not meant to be run standalone during normal use — it's invoked automatically by `retrieve_logic.py` the first time the app runs and no cached index is found (only the caption/text side is *not* used for indexing here; only image embeddings are stored).
 
### `retrieve_logic.py`
The **core retrieval logic** and the module that ties everything together. On import, it:
- Loads the CLIP model and processor (`openai/clip-vit-base-patch16`) once, moving them to GPU if available.
- Loads the Flickr8k dataset (needed to map result indices back to actual images).
- Calls `load_index()`, which checks whether `embeddings.npy` already exists next to this file:
  - if **found**, it loads the cached embeddings directly (fast startup);
  - if **not found**, it calls `build_embeddings()` from `builder_index.py` to compute and save them (slow, first-run only).
It then exposes two functions:
- `get_text_embedding(prompt)` — encodes a text prompt into a normalized 512-dim CLIP embedding.
- `search(prompt, topk)` — computes similarity scores between the prompt embedding and all cached image embeddings, sorts them in descending order, and returns the corresponding top-`k` images from the dataset.
### `embeddings.npy`
The **cached index** of image embeddings (shape `[8000, 512]`), generated automatically the first time the app is run. Deleting this file forces a full re-indexing of the dataset on the next launch.

 
## Running the app
 
From inside the `GradioApp/` folder, with the environment set up and dependencies installed:
 
```bash
python app.py
```
 
Gradio will start a local web server (the URL is printed in the terminal) where you can enter a prompt and browse the retrieved images.
 
> **First run note:** if `embeddings.npy` is not yet present, the app will first download Flickr8k and CLIP, then compute and cache all image embeddings before the interface becomes ready — this can take a while depending on your hardware. Subsequent runs will start almost instantly by reusing the cached `embeddings.npy`.
 
---
 
## Notes & Limitations
 
- Only **image embeddings** are indexed; Flickr8k's ground-truth captions are not currently used (e.g. for evaluation or as additional retrieval signal) 
- The dataset and model are loaded once at import time (module-level globals in `retrieve_logic.py`), so the first request after starting the app may be slightly slower while everything is loaded into memory/GPU.
- Retrieval quality is entirely determined by CLIP's pretrained embedding space — no fine-tuning is performed on Flickr8k in this exercise.
