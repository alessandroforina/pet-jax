# Training a PET model

This example shows how to prepare data and train a PET model.

## 1. Prepare data

Put your dataset as `data.xyz` (extended XYZ format) in this folder, then:

```bash
DATASETS=/path/to/datasets python prepare.py
```

This writes train/valid splits (80/20) to `$DATASETS/my_project/`. Edit `prepare.py`
to change the split ratio, project name, or add extra properties (e.g. stress).

## 2. Configure

Edit `model.yaml` to set the architecture. Key choices:

- `lr: false` — pure short-range PET (faster, no jaxpme Ewald cost)
- `lr: true` — long-range PET (adds Ewald electrostatics)

Edit `settings.yaml` to set the dataset paths, batch size, and training schedule.
Both paths in `settings.yaml` are resolved relative to `$DATASETS`.

## 3. Train

```bash
cd examples/train
DATASETS=/path/to/datasets pet-train
```

Checkpoints, logs, and plots are written to `run/`. Training resumes automatically
if `run/` already exists.

## 4. Load a trained model

```python
from pet.calculator import Calculator

calc = Calculator.from_checkpoint("run/checkpoints/MAE_F")
```
