#!/bin/bash
#SBATCH --job-name=homo_gru_baseline
#SBATCH --account=cpsc4830
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_baseline/seed789/logs/%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_baseline/seed789/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --partition=education_gpu
#SBATCH --gres=gpu:rtx_5000_ada:1


set -euo pipefail

DATA_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4
MODEL_DIR=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model
RUN_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_baseline

mkdir -p ${RUN_DIR}/logs
cd ${MODEL_DIR}

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version

# Copy train_homogeneous_gru_baseline.py into ${DATA_DIR} before running this sbatch,
# or change the path below to wherever you save the script.
python train_homogeneous_gru_baseline.py \
  --data-dir ${DATA_DIR} \
  --out-dir ${RUN_DIR}/seed789 \
  --seed 789 \
  --epochs 20 \
  --batch-size 256 \
  --lr 3e-4 \
  --weight-decay 5e-5 \
  --region-emb-dim 64 \
  --gnn-hidden 128 \
  --gru-hidden 128 \
  --gru-layers 2 \
  --dropout 0.2 \
  --gnn-type sage \
  --num-workers 0 \
  --patience 5 \
  --amp
