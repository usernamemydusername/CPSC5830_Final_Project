#!/bin/bash
#SBATCH --job-name=porto_data_prep
#SBATCH --output=logs/porto_data_prep_%j.out
#SBATCH --error=logs/porto_data_prep_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00

set -euo pipefail

# Edit these paths for your own environment.
PROJECT_DIR=/path/to/your/repository
DATA_DIR=${PROJECT_DIR}/data
OUT_DIR=${DATA_DIR}/trial2

mkdir -p ${OUT_DIR}/logs
mkdir -p logs

cd ${PROJECT_DIR}

# Load modules / activate environment as needed.
# Example:
# module load Python/3.12.3-GCCcore-13.3.0
# source /path/to/your/venv/bin/activate

python data_prep/prepare_data3.py \
  --data-dir ${DATA_DIR} \
  --out-dir ${OUT_DIR} \
  --cell-size 250 \
  --place "Porto, Portugal" \
  --train-frac 0.70 \
  --val-frac 0.15 \
  --num-prefix-samples 5 \
  --chunksize 50000
