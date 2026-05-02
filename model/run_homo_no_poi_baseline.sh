#!/bin/bash
#SBATCH --job-name=homo_gru_no_poi
#SBATCH --account=cpsc4830
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_no_poi_baseline/logs/%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_no_poi_baseline/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:h200:1

set -euo pipefail

DATA_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4
MODEL_DIR=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model
BASE_RUN_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/homo_gru_no_poi_baseline
LOG_DIR=${BASE_RUN_DIR}/logs

mkdir -p ${BASE_RUN_DIR} ${LOG_DIR}
cd ${MODEL_DIR}

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version

for SEED in 123
do
  echo "Running seed ${SEED}"

  python train_homo_no_poi_baseline.py \
    --data-dir ${DATA_DIR} \
    --out-dir ${BASE_RUN_DIR}/seed${SEED} \
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
    --region-feature-mode no_poi \
    --num-workers 0 \
    --patience 5 \
    --seed ${SEED} \
    --amp
done