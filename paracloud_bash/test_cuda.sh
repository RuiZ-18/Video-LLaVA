#!/bin/bash
module purge
source /home/bingxing2/apps/package/pytorch/2.1.0+cuda121_cp310/env.sh
source activate
source activate videollava
export PYTHONUNBUFFERED=1

python test_cuda.py
