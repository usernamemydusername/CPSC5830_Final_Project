#!/bin/bash
#SBATCH --job-name=porto_heterograph_t2
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial2/logs/porto_heterograph_%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial2/logs/porto_heterograph_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=20G
#SBATCH --time=12:00:00

set -euo pipefail

mkdir -p /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial2/logs
cd /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial2

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version
 
python /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/prepare_data3.py \
  --data-dir /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data \
  --out-dir /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial2 \
  --cell-size 250 \
  --place "Porto, Portugal" \
  --train-frac 0.70 \
  --val-frac 0.15 \
  --num-prefix-samples 5 \
  --chunksize 50000