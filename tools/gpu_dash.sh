#!/bin/bash
# Terminal dashboard for the GPU0 eval queue (1 s refresh, stdlib python). Run inside tmux:
#   bash /NHNHOME/share/cjpark/AppCorr-vllm/tools/gpu_dash.sh [chain.sh ...]
# With no argument it follows the newest scratchpad chain_gpu0_*.sh.
exec /home/nxclab/anaconda3/envs/appcorr/bin/python "$(dirname "$0")/gpu_dash.py" "$@"
