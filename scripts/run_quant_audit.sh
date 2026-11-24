#!/usr/bin/env bash
set -euo pipefail
PYTHONPATH=src python3 -m inference_quant_tradeoff_lab.cli audit "$@"
