#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
remote_dir=/home/optomachanics/projects/FlashLLE

ssh -o RemoteCommand=none -o RequestTTY=no optics-cuda \
  "wsl.exe -d Ubuntu-24.04-CUDA -u optomachanics --cd ~ -- mkdir -p $remote_dir"

tar --format ustar --exclude .DS_Store --exclude '._*' \
  --exclude __pycache__ --exclude '*.pyc' --exclude '*.egg-info' \
  -C "$project_dir" -cf - \
  pyproject.toml README.md src tests benchmarks examples scripts \
  | ssh -o RemoteCommand=none -o RequestTTY=no optics-cuda \
    "wsl.exe -d Ubuntu-24.04-CUDA -u optomachanics --cd ~ -- tar -xf - -C $remote_dir"

# The operator implementations moved from operators.py into an operators/
# package. Remove only that obsolete file after the new package is in place.
ssh -o RemoteCommand=none -o RequestTTY=no optics-cuda \
  "wsl.exe -d Ubuntu-24.04-CUDA -u optomachanics --cd ~ -- rm -f $remote_dir/src/flashlle/operators.py"
