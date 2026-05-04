#!/bin/bash
#SBATCH --job-name=hgt_train
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_xc446/hgt_logs/hgt_train_%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_xc446/hgt_logs/hgt_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:1

set -euo pipefail

# Update these paths to match your cluster layout
DATA_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_xc446/porto_data_bundle
CODE_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_xc446
OUT_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_xc446/hgt_results
LOG_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_xc446/hgt_logs

mkdir -p "$OUT_DIR" "$LOG_DIR"

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_xc446/my_env/bin/activate

which python
python --version
nvidia-smi

cd "$CODE_DIR"

python hgt_train.py \
  --data-dir "$DATA_DIR" \
  --out-dir  "$OUT_DIR" \
  --batch-size 256 \
  --seeds 0 1
