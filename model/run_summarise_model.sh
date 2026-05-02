#!/bin/bash
#SBATCH --job-name=summarize_runs
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/summary/logs/%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/summary/logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00


set -euo pipefail

RUN_DIR=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/summary
MODEL_DIR=/home/cpsc4830_ym474/project_cpsc4830/cpsc4830_ym474/model


mkdir -p ${RUN_DIR}/logs
cd ${MODEL_DIR}

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version

python summarize_model_runs.py \
  --model hetero_rgcn_gru=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/model/runs/hetero_group_rgcn_sage_gru/full \
  --seeds 123 456 789\
  --out-dir ${RUN_DIR}
