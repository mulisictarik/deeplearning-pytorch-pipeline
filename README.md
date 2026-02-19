# K2: PyTorch Pipeline & Model Training

Deep Learning pipeline for CIFAR-10 image classification using PyTorch. I built this using `uv` for fast package management and `DVC` to keep track of the dataset.

## Project Structure

```text
deeplearning_pytorch_pipeline/
├── data/                 # CIFAR-10 dataset
├── models/               # Saved model weights (.pth)
├── metrics/              # Saved results from training (.json)
├── src/
│   ├── dataset.py        # Loads and prepares the data
│   ├── model.py          # The CNN model
│   ├── train.py          # Training and validation loops
├── main.py               # Runs all experiments automatically
├── params.yaml           # Settings for the experiments
├── pyproject.toml        # uv configuration
└── README.md             # This file

## Setup

### 1. Install Dependencies

I used `uv` to set up the environment fast. Here are the commands I used to start the project:

uv init
uv add torch torchvision pyyaml tqdm dvc

### 2. Data Versioning (DVC)

The images are tracked with DVC so they dont bloat the Git repository. To download the dataset:

uv run dvc init
uv run dvc add data
git add data.dvc .gitignore
git commit -m "Initialize DVC and track CIFAR-10 dataset"

***Note for reviewers: The repository is already initialized. To download the actual image data to your local machine, simply run: uv run dvc pull

### 3. Run Training

The training is fully automated. Instead of using manual terminal flags, just run `main.py` and it will loop through all experiments in `params.yaml`

uv run python main.py

## Experiment Results

I ran all experiments for 10 epochs to see how the different settings performed.

| Experiment      | Learning Rate | Batch Size | Epochs | Best Accuracy | Notes                                                                                      |
| --------------- | ------------- | ---------- | ------ | ------------- | ------------------------------------------------------------------------------------------ |
| 1 (Baseline)    | 0.001         | 64         | 10     | **70.94%**    | Good stable performance.                                                                   |
| 2 (High LR)     | 0.01          | 64         | 10     | **62.06%**    | Learning rate was too high. The model struggled to learn properly.                         |
| 3 (Large Batch) | 0.001         | 128        | 10     | **72.04%**    | **Best result.** The larger batch size gave more stable updates and overtook the baseline. |

## Conclusion

**Experiment 3 performed best** with 72.04% accuracy.
A larger batch size (128) provided smoother updates over 10 epochs, which helped it find a better result than the baseline (batch size 64). Experiment 2 failed to get good results because an aggressive learning rate of 0.01 makes the model jump around too much and miss the optimal weights.

## Model Architecture

SimpleCNN:

- 3 Convolutional layers (32, 64, 128 filters)
- Max pooling
- 2 Fully connected layers (512, 10)
- Dropout (0.3)

## Dataset

CIFAR-10: 60,000 32x32 color images in 10 classes

- Training: 50,000 images
- Test: 10,000 images

## Reproducibility Checklist

- [x] Dataset tracked with DVC
- [x] Code split into logical modules (`src/`)
- [x] Training runs via `main.py`
- [x] 3 experiments with different hyperparameters
- [x] Uses `uv` for dependencies
```
