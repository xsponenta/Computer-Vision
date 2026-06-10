# Food-101 Image Classification

Computer Vision coursework project for food image classification on a 20-class subset of Food-101. The notebook compares CNN and transformer backbones, explores data cleaning with embedding-based outlier detection, tests progressive unfreezing and learning-rate schedules, runs Weights & Biases experiments, builds an ensemble, and visualizes model attention with Grad-CAM.

## Project Scope

The project uses `torchvision.datasets.Food101` and narrows the experiment to 20 visually diverse food classes. Images are resized to `224 x 224`, normalized, and augmented with modern training transforms such as MixUp/CutMix.

Main experiments:

- exploratory data analysis for class balance and image quality;
- baseline transfer learning with ResNet-50, EfficientNet-B0, and ViT-B/16;
- precision, recall, F1, and accuracy tracking;
- label/noise inspection with Cleanlab and Isolation Forest over embeddings;
- CLIP embedding outlier detection as an alternative cleaning strategy;
- ResNet-18 progressive unfreezing with cosine and linear warmup schedules;
- Weights & Biases hyperparameter sweeps over learning rate, dropout, and weight decay;
- ensemble weighting and temperature scaling with 5-fold validation;
- Grad-CAM comparison for ResNet and EfficientNet.

## Repository Contents

| File | Purpose |
| --- | --- |
| `ivanyshyn_food_101.ipynb` | Main notebook with data loading, experiments, plots, metrics, and conclusions |
| `Ivanyshyn Food Weights & Biases.pdf` | Exported W&B experiment report |
| `requirements.txt` | Python dependencies used by the notebook |

## Method Overview

```text
Food-101 images
  -> select 20 classes
  -> train/validation split
  -> resize + normalize + augmentation
  -> train ResNet / EfficientNet / ViT baselines
  -> audit possible label issues
  -> train improved ResNet variants
  -> W&B sweep
  -> ensemble and temperature scaling
  -> Grad-CAM interpretation
```

## Notable Findings

- The selected Food-101 subset is balanced by construction, but visually similar dishes still create ambiguity.
- ResNet performed best among the initial ResNet/EfficientNet/ViT comparison in this setup.
- Removing suspected noisy samples with Cleanlab/Isolation Forest did not improve final performance, suggesting some removed samples were hard-but-useful examples.
- Progressive unfreezing with warmup schedules improved the ResNet-18 fine-tuning pipeline.
- The ensemble optimizer converged close to equal weights and temperature `1.0`, indicating that the base models were already reasonably calibrated.
- The ensemble improved the second-stage results by about 1.2 percentage points.
- Grad-CAM showed that ResNet focused more tightly on central food regions, while EfficientNet sometimes attended to broader context such as plates, forks, and image corners.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Food-101 is downloaded automatically by `torchvision.datasets.Food101` when the notebook is run with `download=True`.

## Run

Start Jupyter and open the notebook:

```bash
jupyter notebook ivanyshyn_food_101.ipynb
```

Or run it from the command line:

```bash
jupyter nbconvert --to notebook --execute ivanyshyn_food_101.ipynb
```

## Generated Artifacts

The notebook writes model checkpoints and local outputs that are intentionally not committed:

```text
saved_models/
data/
wandb/
```

Re-run the notebook to regenerate these artifacts.

## Notes

- The notebook uses pretrained torchvision models and may download model weights on first run.
- Some sections require W&B authentication for experiment tracking.
- Grad-CAM requires saved model checkpoints from the earlier training cells.
