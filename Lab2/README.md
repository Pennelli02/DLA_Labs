# Lab 2 — The Transformative Transformer

## Overview
This laboratory focuses on gaining hands-on experience with **Transformer models**, mostly through the **Hugging Face ecosystem**. Starting from a sentiment analysis task on movie reviews, the lab progresses from a training-free baseline (feature extraction + SVM) to full fine-tuning of a pretrained Transformer, and finally to a practical application: a **text-to-image retrieval system** built with CLIP.

This lab is split across two locations:
- **Exercises 1 and 2** are implemented in the `DLA-Lab2.ipynb` notebook.
- **Exercise 3** is implemented as a standalone application in the `GradioApp/` folder, since it is explicitly meant to be built outside of a notebook. See [`GradioApp/README.md`](./GradioApp/README.md) for details on how it works.
---

## Repository Structure
 
```
Lab2/
├── DLA-Lab2.ipynb        # Exercises 1 and 2 (Sentiment Analysis + Fine-tuning)
├── GradioApp/              # Exercise 3.3: Text-to-Image Retrieval app (see its own README)
│   ├── app.py
│   ├── builder_index.py
│   ├── retrieve_logic.py
│   ├── embeddings.npy      # cached CLIP image embeddings (auto-generated on first run)
│   └── README.md
└── README.md
```

## Implementations

### Exercise 1
Working with the [Cornell Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) movie review dataset (5,331 positive / 5,331 negative sentences) and a pretrained **DistilBERT** ([`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased)).
 
- **1.1 — Dataset exploration**: loading train/validation/test splits via `datasets`, inspecting label distribution (0 = negative, 1 = positive) and variable sentence length.
- **1.2 — Pretrained BERT & tokenizer**: loading DistilBERT and its tokenizer with `AutoModel`/`AutoTokenizer`, inspecting tokenization/decoding, and understanding the role of the **attention mask** in masking out padding tokens during the forward pass.
- **1.3 — Stable baseline**: using a Hugging Face `feature-extraction` pipeline to extract the `[CLS]` token from DistilBERT's last hidden layer for every sentence, then training a **Linear SVM** (`sklearn.svm.LinearSVC`) on top of these frozen features.
  - A log-spaced grid search over the SVM regularization parameter `C` (`np.logspace(-4, 4, 20)`) is used to select the best value on the validation split, which is then evaluated on the test split.
---
### Exercise 2
- **2.1 — Tokenization**: the dataset is pre-tokenized (once) using `Dataset.map`, producing `input_ids` and `attention_mask` columns alongside the original `text`/`label` columns, then converted to PyTorch tensors.
- **2.2 — Model setup**: `DistilBertForSequenceClassification` is used to attach a new, randomly-initialized classification head on top of the `[CLS]` token for binary sentiment classification.
- **2.3 — Fine-tuning with `Trainer`**: the Hugging Face `Trainer` API is used to fine-tune the full model, with:
  - a `DataCollatorWithPadding` for dynamic batch padding;
  - a `compute_metrics` function reporting accuracy and F1 (via the `evaluate` library);
  - `TrainingArguments` covering optimizer (`AdamW`, `SGD`, fused variants), LR scheduler (`linear`, `cosine`), mixed precision (`fp16`/`bf16`), gradient clipping, and TensorBoard logging.
  - A `finetunepipe` wrapper function was implemented to quickly sweep over multiple training configurations (learning rate, scheduler, optimizer, epochs, batch size) and evaluate each on the test split.
 
**Results:** the model tends to overfit quickly, so the best configurations use around **3–5 epochs**. Across the various hyperparameter combinations tested, accuracy and F1 score consistently land around **83–84%**, a clear improvement over the frozen-feature SVM baseline from Exercise 1.
 
---
### Exercise 3
Implemented as a standalone application (outside the notebook, as suggested by the exercise) in the [`GradioApp/`](./GradioApp/) folder. It indexes the [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k) dataset using **CLIP** image embeddings and serves a **Gradio** web interface where a user can type a text prompt and retrieve the most visually-similar images.
 
See [`GradioApp/README.md`](./GradioApp/README.md) for the full explanation of how the application works, its files, and how to run it.
 
---
## AI Usage Disclosure
 
This notebook made use of **Claude** and **Gemini** mainly as support for fixing code errors, clarifying doubts, and searching for relevant Hugging Face documentation/resources. 
 
