# LAB1 From Pixels to Semantics

## Overview
Lab on classification and retrieve information, using datasets containing street images. First we start with classification using pretrained models and technic of transfer learning (using a backbone trained in another dataset, remove classfication head and add a new one and finetunig). At the end using different feature extractor for information retrieval.

The lab is divided into 3 exercises and everyone can find in **DLA-Lab1.ipynb**.

## Project Structure
```text
Lab1
├─── DLA-Lab1.ipnyb
├───paper_lab1/
├───runs/
|  
└───_data/
```
Where in the folder **paper_lab1** there are the papers where come from the ideas for the third esercise, in **runs** there are the experiments and training log (tensorboard) and in **_data** there is the dataset.

## Implemtations
### Exercise 1 
In this exercise, we first perform an exploration of the data (Exploratory Data Analysis) on the GTSRB dataset. Next, we use a pre-trained ResNet18 to perform feature extraction. The features extracted in this way are used to train and evaluate an SVM. SVM like classification head obtain an accuracy about 63%. Instead using finetunig in different backbones like resnet18 obtain an accuracy about 90%

---
### Exercise 2
In this exercise we do a refactor and added more generalitation on in this case of classifications managed by a parmeters cell. 
Main components are:
- **Configuration management** — a single `config` dictionary (converted to a `SimpleNamespace`) centralizes all hyperparameters (model name, image sizes, dataset split, optimizer, scheduler, loss, training/checkpointing options), replacing scattered global variables.
- **Dataloaders** — `MakeDataloaders` builds a reproducible train/validation split (`torch.utils.data.random_split` with a fixed seed) and wraps it into `DataLoader`s.
- **Model abstraction** — `get_pretrained_model` loads any `torchvision` model, optionally freezes all or specific layers, and dynamically replaces its head with a small MLP sized to the number of classes. `get_personal_model` builds a lightweight custom CNN (`PersonalCNN`) as an alternative, non-pretrained architecture.
- **Optimizer / scheduler abstraction** — the training loop supports `SGD`, `Adam`, and `AdamW` optimizers (with sensible default weight decay) and an `ExponentialLR` scheduler, all driven by the configuration object.
- **Logging** — training runs are logged with **TensorBoard** (`SummaryWriter`), including hyperparameters as text, one log directory per run under `runs/<model_name>_<optimizer>_lr<lr>`. A [Weights & Biases](https://wandb.ai/site/) integration was attempted first but was replaced by TensorBoard due to integration issues encountered during development.

To visualize TensorBoard logs from the command line:
 
```bash
tensorboard --logdir Lab1/runs
```
---
### Exercise 3
This exercise reframes traffic-sign classification as an **image retrieval** problem: the training set acts as a *gallery* of indexed descriptors, and the test set provides *query* descriptors, avoiding the need for any fine-tuning.
 
**Pipeline overview:**
1. **Feature extraction** — a generic `get_feature_extractor` function supports multiple pretrained backbones:
   - **VGG-11** (feature extraction at an intermediate layer)
   - **ResNet-50**, optionally extracting from intermediate residual blocks (`layer1`/`layer2`/`layer3`) instead of the final layer, combined with **GeM pooling** (Generalized Mean Pooling, `p=3`, as proposed in [3])
   - **ViT-B/16**, using the `[CLS]` token
   - **DINOv2** (ViT-B/14, loaded via `torch.hub`), using the `[CLS]` token, chosen based on findings in [4]
   Each backbone has its own dedicated preprocessing transform (`define_transform`), matching the resolution/normalization it was originally trained with.
2. **Post-processing** — extracted features are L2-normalized; optional **PCA whitening** (à la Radenović et al. [2]) can be applied to decorrelate and optionally compress the descriptors.
3. **Similarity & retrieval** — cosine similarity between L2-normalized query and gallery features is computed as a simple matrix product.
4. **Evaluation** — retrieval quality is assessed with per-class Precision-Recall curves, per-class and mean Average Precision (mAP), and cross-checked against the [`pytorch-metric-learning`](https://kevinmusgrave.github.io/pytorch-metric-learning/) library's `AccuracyCalculator` (precision@1, mAP, mAP@R) to validate the custom implementation.
5. **Classification** — a **Nearest-Mean Classifier (NMC)** is built on top of the retrieval pipeline: per-class centroids are computed from the (L2-normalized) gallery features, and each query is assigned to the class of its nearest centroid by cosine similarity.
**Observations:**
- The GTSRB training set is imbalanced (between 150 and 1500 images per class), which is expected to affect similarity-based performance more for under-represented classes.
- The chosen backbones, pooling strategy (GeM), normalization, and evaluation metrics were directly inspired by the reference papers listed below.

#### Results
We tested a lot models with differents configurations so it is difficult to show here every results but you could see launching the tensorboard. 
In general we could see:
**key findings**
- **VGG-11** is the best-performing backbone across all retrieval metrics, and after PCA compression to 512 dimensions its Nearest-Mean Classifier reaches **~90% accuracy/mAP** — by far the best result obtained, and not matched by any other backbone.
- **ResNet-50** benefits substantially from **GeM pooling** over standard average pooling, from higher input resolution (224×224 vs. 64×64), and from extracting features at an **intermediate layer** (layer 3) rather than the final one.
- **ViT-B/16** and **DINOv2** perform poorly in this retrieval setting (mAP ≈ 13–14%), despite DINOv2 being reported as state-of-the-art for feature extraction in the reference paper. This is likely due to GTSRB's small, highly variable image sizes: patch-based transformers require fixed input resolutions, and the resizing/upscaling needed to fit their expected input likely degrades the images too much to preserve useful information.
- **PCA whitening** tends to reduce raw retrieval accuracy and mAP (especially for ViT, DINOv2, and ResNet's last/penultimate layers), but consistently **improves downstream NMC classification** — likely because it removes noise while retaining the essential information needed for classification, at the cost of some fine-grained variability between similar signs.
- Classes **12** (priority road), **13** (yield), and **14** (stop) are consistently the easiest to identify across all backbones, likely due to a combination of larger class representation in the dataset and their distinctive shape/color compared to other signs.
---
## References
 
The following papers, collected in `paper_lab1/`, guided the design of Exercise 3.2:
 
1. Houben, S., Stallkamp, J., Salmen, J., Schlipsing, M., Igel, C. *Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark.* IJCNN 2013. (Dataset source.)
2. [Neural Codes for Image Retrieval](https://arxiv.org/abs/1404.1777) — informed the use of CNN (VGG) features for retrieval, the evaluation metrics (accuracy, mAP, per-class mAP), and the idea of using intermediate ResNet-50 layers.
3. [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512) — source of the GeM pooling (`p=3`) used to improve retrieval performance.
4. [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193) — motivated the choice of DINOv2 as a backbone and the use of cosine similarity with L2-normalization.
5. [All You Need to Know About Training Image Retrieval Models](https://arxiv.org/abs/2503.13045) — systematic study of factors affecting image retrieval training performance.
---
 
## AI Usage Disclosure
 
AI assistance was used during this lab for an initial research and brainstorming phase — in particular, to help identify relevant papers on image retrieval before Exercise 3.2 was implemented.

