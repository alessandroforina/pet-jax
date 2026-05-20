#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-/path/to/checkpoint}"
SOCKET_NAME="pet"

# Install PET driver into i-PI (idempotent)
pet-install-ipi-driver

# Start i-PI server in background
i-pi input.xml > i-pi.out 2>&1 &
IPI_PID=$!
echo "Started i-PI server (PID $IPI_PID)"

# Give the server time to open the socket
sleep 5

# Start PET driver
i-pi-driver -a ${SOCKET_NAME} -u -m pet \
    -o model_path=${MODEL_PATH},template=start.xyz \
    > driver.out 2>&1 &
DRIVER_PID=$!
echo "Started PET driver (PID $DRIVER_PID)"

# Wait for both processes
wait $IPI_PID $DRIVER_PID
echo "Simulation complete."
