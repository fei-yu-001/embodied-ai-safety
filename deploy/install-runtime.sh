#!/usr/bin/env bash
set -euo pipefail
root="${PRISM_INSTALL_ROOT:-/opt/prism}"
sudo apt-get install -y python3-venv python3-dev build-essential linux-libc-dev libosmesa6 libgl1 libglib2.0-0 libglfw3
sudo install -d -o "$USER" -g "$USER" "$root" "$root/shared" "$root/releases"
python3 -m venv "$root/shared/venv"
"$root/shared/venv/bin/pip" install --upgrade pip wheel setuptools
"$root/shared/venv/bin/pip" install -r server/requirements.txt -r deploy/requirements-simulation.txt
mkdir -p "$root/shared/vendor" "$root/shared/sessions" "$root/shared/cache"
if [[ ! -d "$root/shared/vendor/robocasa/.git" ]]; then
    git clone --filter=blob:none https://github.com/robocasa/robocasa.git "$root/shared/vendor/robocasa"
fi
if [[ ! -d "$root/shared/vendor/robosuite/.git" ]]; then
    git clone --filter=blob:none https://github.com/ARISE-Initiative/robosuite.git "$root/shared/vendor/robosuite"
fi
git -C "$root/shared/vendor/robocasa" checkout 3443cdf3dd20f9ae8ed41adba442059b506891d0
git -C "$root/shared/vendor/robosuite" checkout 5ce6643f3092639d08f7b0f90ed1c6a84f50552c
# Restore separately downloaded assets before starting a real simulation.
echo "Runtime ready. Install RoboCasa assets, build web/, and configure prism.env before starting."
