# PET-JAX

JAX implementation of PET (Point Edge Transformer), a machine learning interatomic potential based on edge-to-edge attention with optional long-range Ewald interactions.

Built on [JAX](https://github.com/jax-ml/jax), [Flax](https://github.com/google/flax), [e3x](https://github.com/google-research/e3x), [jax-pme](https://github.com/lab-cosmo/jax-pme), and [marathon](https://github.com/sirmarcel/marathon).

## Installation

Requires Python >= 3.11.

```bash
pip install .
```
On Kuma cluster it may be needed to execute the following commands due to compatibility issues that may arise with newest version of jax.

```bash
pip install "flax<0.12" --break-system-packages
pip install jax==0.9.2 jaxlib==0.9.2 jax-cuda12-pjrt==0.9.2 jax-cuda12-plugin==0.9.2 --break-system-package
pip install nvidia-cudnn-cu12==9.8.0.87 --break-system-packages
```
Before running the job it may be useful to export:
```bash
export LD_LIBRARY_PATH=$(python -c "
import os, nvidia
base = os.path.dirname(nvidia.__file__)
libs = [os.path.join(base, d, 'lib') for d in os.listdir(base) if os.path.exists(os.path.join(base, d, 'lib'))]
print(':'.join(libs))
"):$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1
python -c "import jax; print(jax.devices())"
```
## Usage

### ASE calculator

```python
import jax
from ase.build import bulk
from pet.model import LongRangePET
from pet.calculator import Calculator

model = LongRangePET(cutoff=5.0)
params = model.init(jax.random.key(42), *model.dummy_inputs())
calc = Calculator.from_model(model, params=params)

atoms = bulk("Ar") * [2, 2, 2]
calc.calculate(atoms)
print(calc.results["energy"], calc.results["forces"].shape)
```

To load a trained model from a checkpoint:

```python
calc = Calculator.from_checkpoint("path/to/checkpoint")
```

### Model variants

**`LongRangePET`** supports two modes via the `lr` flag:

| Setting | Description |
|---|---|
| `lr=False` (default) | Pure short-range PET: edge transformer only |
| `lr=True` | Long-range PET: edge transformer + Ewald electrostatics |

### Key hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `cutoff` | 5.0 | Neighbor list cutoff radius (Å) |
| `lr` | False | Enable long-range Ewald interactions |
| `num_hidden` | 128 | Hidden feature dimension |
| `num_hidden_feedforward` | 256 | Feedforward network width |
| `num_attention_layers` | 2 | Transformer layers per message-passing step |
| `num_message_passing_layers` | 2 | Number of message-passing iterations |
| `num_heads` | 4 | Attention heads |
| `cutoff_width` | 0.2 | Cosine cutoff onset width (Å) |
| `num_charges` | 8 | Number of learnable charge channels (lr only) |

### Training

Training uses [marathon](https://github.com/sirmarcel/marathon) and follows the same
workflow as other marathon-based models.

#### 1. Prepare data

```python
from marathon.data import datasets
from marathon.grain import prepare

prepare(train_atoms, folder=datasets / "my_project/train")
prepare(valid_atoms, folder=datasets / "my_project/valid")
```

#### 2. Configure experiment

Each experiment lives in its own directory with two YAML files.

**`model.yaml`:**
```yaml
model:
  pet.LongRangePET:
    cutoff: 5.0
    lr: true
    num_hidden: 128
    num_message_passing_layers: 2

baseline:
  elemental:
    1: -3.7
    8: -7.0
```

**`settings.yaml`:**
```yaml
train: "my_project/train"
valid: "my_project/valid"
batcher:
  batch_size: 4
loss_weights: {"energy": 0.5, "forces": 0.5}
optimizer: adam
start_learning_rate: 1e-3
max_epochs: 2000
use_wandb: true
```

#### 3. Run training

```bash
cd my_experiment
DATASETS=/path/to/datasets pet-train
```

## Installing the i-PI driver

After installing the package, install the i-PI driver via:

```bash
pet-install-ipi-driver
```

This copies the PET driver into the i-PI `pes` directory. Rerun anytime you reinstall the package or switch environments.

### Running MD with i-PI

```bash
# Start i-PI server
i-pi input.xml &
sleep 5

# Start PET driver
i-pi-driver -a pet -u -m pet \
    -o model_path=/path/to/checkpoint,template=start.xyz
```

See `examples/md-ipi/` for a complete example with `input.xml` and `run.sh`.

## Development

Format and lint:

```bash
ruff format . && ruff check --fix .
```

Run tests:

```bash
python -m pytest tests/ -v --override-ini="addopts="
```

## License

BSD-3-Clause
