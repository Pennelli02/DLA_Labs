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
### Exercise 2
In this exercise we do a refactor and added more generalitation on in this case of classifications managed by a parmeters cell
### Exercise 3
In this exercise we test differents models on information retrieval task and implemented a NMC for classification. 
All tests could be see from tensorboard:
```bash
tensorboard --logdir=Lab1\runs
```
