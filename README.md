# PyTorch Pipeline – CIFAR-10 Image Classification

A deep learning pipeline that trains a CNN to classify images from CIFAR-10. Experiments are defined in `params.yaml`, run automatically one after another, and every run saves its best model and its metrics. The dataset is versioned with DVC so it stays out of Git.

## Tech stack

Python · PyTorch · torchvision · DVC · uv · YAML

## Results

All experiments ran for 10 epochs.

| Experiment | Learning rate | Batch size | Best accuracy | Notes |
|---|---|---|---|---|
| 1 – Baseline | 0.001 | 64 | 70.94 % | Stable training |
| 2 – High LR | 0.01 | 64 | 62.06 % | Learning rate too high, the model never settled |
| 3 – Large batch | 0.001 | 128 | **72.04 %** | Best result, smoother updates than the baseline |

A larger batch size gave the best result, while a learning rate of 0.01 made the model jump past good weights.

## Run it

```bash
uv sync
uv run dvc pull          # download the dataset
uv run python main.py    # runs every experiment in params.yaml
```

Models are saved to `models/` and metrics to `metrics/`.

## Project structure

```text
├── src/
│   ├── dataset.py   # loading and transforming the data
│   ├── model.py     # CNN architecture
│   └── train.py     # training and validation loops
├── main.py          # runs all experiments
├── params.yaml      # experiment settings
├── models/          # best weights per experiment (.pth)
├── metrics/         # results per experiment (.json)
└── data.dvc         # DVC pointer to the dataset
```
