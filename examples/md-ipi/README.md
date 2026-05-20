# MD with PET via i-PI

This example shows how to run NVT molecular dynamics using a trained PET model and i-PI.

## Prerequisites

- Trained PET checkpoint (with `model/model.yaml`, `model/baseline.yaml`, `model/model.msgpack`)
- i-PI installed (`pip install ipi`)
- PET i-PI driver installed: `pet-install-ipi-driver`

## Files

- `input.xml` — i-PI configuration for NVT-MD at 300 K. Adapt the cell, thermostat, timestep, and starting geometry to your system.
- `start.xyz` — Starting structure in atomic units. Replace with your system's geometry.
- `run.sh` — Launch script. Set `MODEL_PATH` to your trained checkpoint folder.

## Running

```bash
MODEL_PATH="/path/to/checkpoint"

# Install PET driver into i-PI (idempotent)
pet-install-ipi-driver

# Start i-PI server
i-pi input.xml > i-pi.out &
sleep 5

# Start PET driver
i-pi-driver -a pet -u -m pet \
    -o model_path=${MODEL_PATH},template=start.xyz \
    > driver.out &

wait
```

See `run.sh` for a self-contained reference script.

i-PI outputs are written to `i-pi.*` files:
- `i-pi.properties.out` — energy, temperature, conserved quantity over time
- `i-pi.positions_*` — nuclear trajectories

## Adapting to your system

1. Replace `start.xyz` with your starting geometry (atomic units).
2. Set the cell in `input.xml` (must match your system).
3. Set `pbc='True'` in `<ffsocket>` if your system is periodic.
4. Adjust timestep and temperature to your target conditions.
5. Point `MODEL_PATH` in `run.sh` to your trained checkpoint.
