# Lab 4 — OOD Detection and Adversarial Robustness
## Overview
This laboratory develops a methodology for **Out-of-Distribution (OOD) detection** and for measuring its quality, and then experiments with **adversarial attacks** (FGSM) and **adversarial training** to make a model more robust.

The lab is divided into 3 exercises and everyone can find in **Lab4_OOD.ipynb**.
---
 
## Repository Structure
 
```
Lab4/
├── Lab4_OOD.ipynb        # All exercises (1, 2, 3)
├── resnet18_best.pth      # Best checkpoint: fine-tuned ResNet-18 on CIFAR-10
├── simplecnn_best.pth     # Best checkpoint: custom SimpleCNN trained on CIFAR-10
├── adv_cnn_best.pth       # Best checkpoint: SimpleCNN trained with adversarial augmentation
├── data/                  # CIFAR-10 / CIFAR-100 (auto-downloaded by torchvision)
├── runs/                  # TensorBoard logs, one subfolder per experiment
└── README.md
```
 
**Note:** `data/` and `runs/` are generated automatically the first time the notebook is executed.

To skip re-training from scratch, the notebook can load the provided checkpoints (`resnet18_best.pth`, `simplecnn_best.pth`, `adv_cnn_best.pth`) directly via the `load_checkpoint()` utility.
 
---
## Implementations
### Exercise 1
**In-Distribution (ID)** dataset: **CIFAR-10**. **Out-of-Distribution (OOD)** dataset: a curated **far-OOD subset of CIFAR-100**, keeping only classes with no semantic overlap with CIFAR-10 (flowers, containers, and tree species — e.g. `orchid`, `bottle`, `oak_tree`), to ensure a clean separation between ID and OOD concepts.
 
Three classification models were fine-tuned/trained on CIFAR-10 to later be evaluated as OOD detectors:
- **ViT-B/16** (`torchvision`, pretrained) — attempted but **discarded**: fine-tuning took considerably longer than the other models and only reached ~45% test accuracy.
- **ResNet-18** (`torchvision`, pretrained, fine-tuned with a custom MLP head) — best classifier, reaching **~90% test accuracy**. The confusion matrix shows the model regularly confuses *dog* and *cat*.
- **SimpleCNN** — a custom, from-scratch CNN (4 conv blocks + BatchNorm + global average pooling + MLP head), reaching **~78% test accuracy**.
Given the poor cost/benefit ratio of ViT, only **ResNet-18** and **SimpleCNN** were carried forward for OOD detection.
 
**OOD scoring functions** implemented:
- `max_logit` — the maximum raw logit value.
- `max_softmax` — the maximum softmax probability (with a configurable temperature).
Qualitative inspection tools were built to support the analysis: per-sample logits/softmax bar plots alongside the input image, percentile-aligned "sorted score" curves comparing ID vs. OOD, and overlaid score-distribution histograms.
 
**Observations:** both models visually separate ID and OOD score distributions to some extent, but the overlap is non-trivial — ResNet-18 shows less separation (i.e. is more *overconfident* on OOD inputs) than SimpleCNN.

Detection quality was measured with the **AUROC** (area under the ROC curve), treating ID-vs-OOD as a binary discrimination problem, following the evaluation approach used in the [ODIN paper](https://arxiv.org/pdf/1706.02690.pdf).
 
| Model | Score function | AUROC |
|---|---|---|
| ResNet-18 (fine-tuned) | max_logit | **91%**|
| ResNet-18 (fine-tuned) | max_softmax | **87%** |
| **SimpleCNN** (from scratch) | max_logit | **93%** |
| **SimpleCNN** (from scratch) | max_softmax | **89%** |
 
**Key finding:** SimpleCNN discriminates ID vs. OOD slightly better than the fine-tuned ResNet-18. This is likely because SimpleCNN was trained *only* on CIFAR-10, while ResNet-18 started from ImageNet-pretrained weights and may have encountered visually similar images to the OOD classes during pretraining, making it comparatively more confident (and less discriminative) on the chosen OOD set.

---
### Exercise 2

The **Fast Gradient Sign Method (FGSM)** was implemented for both **untargeted** attacks (push the prediction away from the ground-truth class) and **targeted** attacks (push the prediction toward a specific target class), applied iteratively until the attack succeeds or a maximum number of iterations is reached. All experiments in this exercise focus on **SimpleCNN** (the approach generalizes to other models with minor adjustments).
 
**Observations:** untargeted attacks require less perturbation budget than targeted ones, and — as expected — the required budget for a targeted attack depends on the target class (e.g. attacking towards "dog" needed a different budget than towards "truck").

The training loop was extended to optionally mix **clean and adversarially-perturbed (FGSM) samples** on-the-fly during training (each batch partially replaced with fresh adversarial examples generated from the *current* model state), in an attempt to improve robustness.
 
**Results:** contrary to the goal, adversarial augmentation **did not improve** robustness in this setup:
- Test accuracy **decreased** (≈79% → ≈77%).
- OOD detection AUROC also **decreased** (≈93% → ≈89%).
This is likely because the attacks primarily targeted the ID dataset, reducing the model's confidence/calibration on in-distribution classification itself, and the hyperparameters used were not tuned specifically to bring out the benefits of adversarial training.
 
---

### Exercise 3

A batched version of the targeted FGSM attack was implemented (`FGSM_batch`) and evaluated systematically:
- For every CIFAR-10 class as the attack target, the **Attack Success Rate (ASR)** and **robust accuracy** were measured across the test set (excluding samples already misclassified or already belonging to the target class).
- The experiment was repeated across a range of perturbation budgets `epsilon` (from 3/255 to 16/255) to study the trade-off between attack strength and perturbation visibility.
**Observations:** as `epsilon` increases, the **attack success rate increases** while the **model's robust accuracy decreases** — but larger perturbations also become **visibly more noticeable**, making the attack easier to detect by simple visual inspection.
 
---

## Model Checkpoints
 
| File | Description |
|---|---|
| `resnet18_best.pth` | Best fine-tuned ResNet-18 checkpoint (CIFAR-10 classification), used in Exercise 1. |
| `simplecnn_best.pth` | Best SimpleCNN checkpoint (standard training, no adversarial samples), used in Exercises 1 and 2.1/2.2 comparisons. |
| `adv_cnn_best.pth` | Best SimpleCNN checkpoint trained **with** adversarial augmentation (Exercise 2.2). |
 
All checkpoints can be loaded with the notebook's `load_checkpoint(model, checkpoint_path, device)` utility, which loads the saved `state_dict` and sets the model to `eval()` mode.
 
---

## References
 
- Liang, S., Li, Y., Srikant, R. [*Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks (ODIN)*](https://arxiv.org/pdf/1706.02690.pdf) — informed the AUROC/AUPR-based evaluation approach for OOD detection (Exercise 1.2).

## AI Usage Disclosure
 
**Claude** was used during this lab mainly for brainstorming, gathering background information/ideas, and assisting with plotting/visualization code.
