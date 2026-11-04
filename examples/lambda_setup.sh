#!/bin/bash
# Prepare a fresh Lambda Cloud GPU instance (Ubuntu) to run the vLLM examples.
#   bash examples/lambda_setup.sh
# Run it from the repository root on the instance. It installs bubblewrap, tmux and uv,
# then builds .venv with the vLLM and example extras.
set -euo pipefail
cd "$(dirname "$0")/.."

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

sudo apt-get update -qq
sudo apt-get install -y -qq bubblewrap tmux

# Ubuntu 24.04 blocks the unprivileged user namespaces bubblewrap needs. Allow them on this machine.
if [[ "$(sysctl -n kernel.apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)" == "1" ]]; then
  echo "kernel.apparmor_restrict_unprivileged_userns=0" | sudo tee /etc/sysctl.d/60-bwrap-userns.conf >/dev/null
  sudo sysctl -q -p /etc/sysctl.d/60-bwrap-userns.conf
fi
bwrap --ro-bind / / --dev /dev --proc /proc --unshare-all true && echo "bubblewrap works"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[examples,vllm,dev]"
.venv/bin/python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'vllm', vllm.__version__)"
