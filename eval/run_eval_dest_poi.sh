#!/bin/bash
#SBATCH --job-name=eval_dest_poi
#SBATCH --account=cpsc4830
#SBATCH --output=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model/analyses/dest_poi/logs/%j.out
#SBATCH --error=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model/analyses/dest_poi/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --partition=day

set -euo pipefail

DATA_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4
MODEL_DIR=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model
ANALYSIS_DIR=${MODEL_DIR}
OUT_DIR=${ANALYSIS_DIR}/dest_poi
CHECKPOINT=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model/runs/hetero_group_rgcn_sage_gru/full/seed123/best_hetero_rgcn_gru.pt

mkdir -p ${OUT_DIR}/logs
cd ${ANALYSIS_DIR}

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version

python eval_dest_poi_groups.py \
  --data-dir ${DATA_DIR} \
  --checkpoint ${CHECKPOINT} \
  --out-dir ${OUT_DIR} \
  --split test \
  --batch-size 256 \
  --num-workers 0 \
  --edge-set full \
  --hetero-conv-type sage \
  --region-feature-mode full \
  --poi-feature-mode grouped \
  --poi-group-mapping ${DATA_DIR}/poi_group_mapping.json
