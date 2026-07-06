# DLA Labs
This repository contains the lab exercises for the Deep Learning Applications course. 
The Labs in particular are:
- From Pixels to Semantics (**Lab1**)
- The Transformative Transformer (**Lab2**)
- OOD Detection and Adversarial Robustness (**Lab4**)

## Enviroment Setup
To run notebooks and script, simply clone the repository and and set up a virtual python environment:
```bash
git clone https://github.com/Pennelli02/DLA_Labs.git
cd DLA_Labs
```
```bash
python -m venv .venv
source .venv/bin/activate
```
On Windows
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```
> **Note:** PyTorch is listed without a pinned CUDA build. Install it first following the instructions at https://pytorch.org/get-started/locally/ to match your system's CUDA version, then install the remaining requirements.
> For an exact reproduction of the development environment (including the CUDA build used, `cu128`), see `requirements-freeze.txt`.

> Requires Python 3.10+
Once activated, you can run the notebooks with Jupyter or your preferred IDE (e.g. VS Code), making sure the `.venv` kernel is selected as the interpreter.

## Repo Structure
The repository is organized into three subfolders, one for each lab:
- Lab1
    -**DLA-Lab1.ipynb**: A notebook containing the solutions to all the exercises.
- Lab2
    -**DLA-Lab2.ipynb**: A notebook containing the solutions of first and second exercises
    -**GradioApp**: a folder containing the solutions of third exercise
- Lab4
    -**DLA-Lab4.ipynb**: A notebook containing the solutions to all the exercises.
  
Each of the three subfolders has a dedicated README.md file that explains the exercise in detail.
