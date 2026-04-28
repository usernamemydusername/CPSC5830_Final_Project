#!/bin/bash
#SBATCH --job-name=porto_region_priors
#SBATCH --output=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4/logs/porto_region_priors_%j.out
#SBATCH --error=/nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4/logs/porto_region_priors_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail

mkdir -p /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4/logs
cd /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4

module load Python/3.12.3-GCCcore-13.3.0
source /home/cpsc4830_ym474/my_env/bin/activate

which python
python --version

python /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/build_region_coord_priors.py \
  --bundle /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4/porto_data_bundle_trial3.tar.gz \
  --out /nfs/roberts/project/cpsc4830/cpsc4830_ym474/data/trial4/region_coord_priors.pt \
  --alphas 0 5 10 20