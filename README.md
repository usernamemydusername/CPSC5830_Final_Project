# Context-Aware Taxi Destination Prediction via Heterogeneous Urban Graphs
A final project for CPSC5830.

**Group Member:** Charles Cai & Yidan Mei

## 0. Environment Setup

Experiments were run on the Yale cluster using SLURM with Python 3.12.3 and `uv` for environment management. To reproduce the environment, load the Python module, create/activate a virtual environment, and install the recorded dependencies:

```bash
module load Python/3.12.3-GCCcore-13.3.0
uv venv my_env
source my_env/bin/activate
uv pip install -r requirements.txt
```
The main dependencies include PyTorch 2.5.0 with CUDA 12.1, PyTorch Geometric 2.7.0, torch-scatter, torch-cluster, GeoPandas, OSMnx, pandas, NumPy, and scikit-learn. Exact package versions are listed in `requirements.txt`.

## 1. Raw data

This project uses the Porto taxi trajectory dataset from the ECML/PKDD 2015 Taxi Trajectory Prediction challenge. The preprocessing script expects the raw Kaggle files to be placed in the data directory as:

```text
data/
├── train.csv
└── test.csv
```

The file `train.csv` is required for constructing the supervised training, validation, and test splits. The file `test.csv` is used to construct unlabeled Kaggle test prefixes when available.

The raw dataset can be downloaded using KaggleHub. This requires Kaggle authentication and access to the competition data.

```python
import kagglehub
import zipfile
import os
import shutil

# Download Kaggle competition data.
data_dir = kagglehub.competition_download(
    "pkdd-15-predict-taxi-service-trajectory-i"
)
print("Downloaded to:", data_dir)

# Unzip downloaded files if needed.
for f in os.listdir(data_dir):
    if f.endswith(".zip"):
        zip_path = os.path.join(data_dir, f)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)

# Copy raw files to the project data directory.
project_raw = "data" # you can change this to your data directory
os.makedirs(project_raw, exist_ok=True)

for name in ["train.csv", "test.csv"]:
    src = os.path.join(data_dir, name)
    dst = os.path.join(project_raw, name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Could not find {src}")
    shutil.copy2(src, dst)

print("Files in project data directory:", os.listdir(project_raw))
```

## 2. Data Preparation

The preprocessing script is located at `data_prep/prepare_data4.py`. It takes the raw Porto taxi files, `train.csv` and optionally `test.csv`, and converts them into the processed data bundle used by our destination prediction task. The script builds prefix-to-destination supervised examples, constructs a heterogeneous urban graph from trajectory transitions, OpenStreetMap roads, and OpenStreetMap POIs, and saves the processed outputs as a compressed `.tar.gz` bundle.

The expected raw data layout is:

After running the preprocessing script, the main output is:

```text
data/trial4/porto_data_bundle_trial3.tar.gz
```

This bundle contains the processed heterogeneous graph, ID mappings, feature names, preprocessing summary, Kaggle test prefixes, and sharded supervised train/validation/test examples. This `.tar.gz` file is the data artifact used by the downstream modeling code.

A SLURM example script is provided in `run_prepare_data4.sh`. It contains the command for running `data_prep/prepare_data4.py`, but users should edit the paths before running it. In particular, update the raw data directory, output directory, log directory, Python environment, and any cluster-specific resource settings. The default paths inside `prepare_data4.py` may also need to be changed or overridden through command-line arguments such as `--data-dir` and `--out-dir`.

Example command:

```bash
python data_prep/prepare_data4.py \
  --data-dir data \
  --out-dir data/trial4 \
  --cell-size 250 \
  --place "Porto, Portugal" \
  --osm-date "2014-06-30T23:59:59Z" \
  --train-frac 0.70 \
  --val-frac 0.15 \
  --num-prefix-samples 5 \
  --chunksize 50000
```


**The resulting `.tar.gz` and `region_coord_priors.pt` files can be found here: https://drive.google.com/drive/folders/1ydVgiwBgh97HlYEWVsgMmZPN2Cj6fYJA?usp=drive_link**.
You can also know more about the data do simple exploratory analyses using `trial2_data_readme.ipynb`.

### Pre-training setup

After extracting the bundle, run the following script once before any training:

```bash
python build_centroids.py
```

This scans all training shards and writes three files into `porto_data_bundle/`:

| File | Purpose |
|---|---|
| `cell_centroids.pt` | Maps each region ID to a GPS centroid `(lat, lon)`, used to compute Haversine distance at evaluation time |
| `taxi_id_map.pt` | Remaps raw taxi IDs to contiguous 0-based indices for use in `nn.Embedding` |
| `gru_param_config.json` | Stores exact model constructor parameters (`num_regions`, `num_dest_classes`, `num_taxi_ids`) derived from the full training set; loaded by all training notebooks so no values are hardcoded |

Load `gru_param_config.json` at the top of any training notebook to pass the correct values to the model constructor:

```python
import json
import torch
from pathlib import Path

DATA_DIR = Path('porto_data_bundle')

with open(DATA_DIR / 'gru_param_config.json') as f:
    cfg = json.load(f)
# cfg = {'num_regions': 6750, 'num_dest_classes': 4978, 'num_taxi_ids': 438}

taxi_id_map = torch.load(DATA_DIR / 'taxi_id_map.pt')
centroids   = torch.load(DATA_DIR / 'cell_centroids.pt')

model = GRUDestinationModel(
    num_regions      = cfg['num_regions'],
    num_dest_classes = cfg['num_dest_classes'],
    num_taxi_ids     = cfg['num_taxi_ids'],
)
```

## 3. Experiments
### Baselines
1. The Markov baseline is implemented in `markov.ipynb`. It predicts the destination cell from the last cell of the prefix using empirical transition counts from the training set, with no learned parameters. For cells unseen in training, it falls back to the global destination frequency distribution.
2. The MLP baseline is implemented in `mlp.ipynb`, with the model defined in `mlp_model.py`. It encodes each prefix as a fixed-length feature vector built from the first and last region in the prefix, concatenated with metadata embeddings, and passed through a 2-layer MLP to predict the destination cell. No recurrent encoder or graph is used.
3. Baseline model of GRU encoded homogeneous graph model is included in the `train_homogeneous_gru_baseline.py`. An example slurm script of submitting the job is also included. It performs message passing on a homogeneous graph whose edges are historical taxi transitions, and then feeds the resulting region embeddings into the GRU trajectory encoder. It uses the same region-level features as the heterogeneous model, but removes explicit POI nodes, road nodes, and heterogeneous edge types.

### Methods

1. GRU encoded heterogeneous R-GCN is included in `train_heterogeneous_rgcn_gru.py`. An example slurm script of submitting the job is also included.
   * Since the raw POI features are sparse and high-dimensional, we further group the POI features manually using `data/poi_group_mapping.json` and then train the grouped heterogeneous R-GCN model. The corresponding code is included in train_heterogeneous_group_rgcn_gru.py`.
   * Ablation analyses: we do the following ablation analyses:
     1) Exclude all poi information and run GRU encoded homogeneous graph (`train_homo_no_poi_baseline.py`).
     2) Set `--edge-set region_features_only` when running `train_heterogeneous_group_rgcn_gru.py` to see whether message passing contributes to the model performance. In this setting, the model does not use any graph edges or message passing. It only projects each region's static features into a region embedding before feeding the prefix sequence to the GRU.
     3) Optional: set `--edge-set no_road` to exclude road nodes and road-related edges. This tests whether road-network information contributes to performance.
     4) Optional: set `--edge-set no_poi` to exclude POI nodes and POI-related edges from the heterogeneous graph. This tests whether explicit POI nodes provide additional benefit beyond region-level features.
     5) Optional: set `--edge-set taxi_only` to keep only taxi-transition edges between regions. This tests whether the heterogeneous model's performance mainly comes from historical mobility transitions rather than POI or road context.
   * For multple-seed runs, one can repeat training with different seed values by setting SEED values. i.e.,
     ```bash
     # Detailed command can be found in 'model/run_hetero_group_rgcn_gru_multiseed.sh'
     for SEED in 123 456 789 {whatever integer seed you like}
     do
       python train_heterogeneous_group_rgcn_gru.py \
           --data-dir ${DATA_DIR} \
           --out-dir ${BASE_RUN_DIR}/seed${SEED} \
           --seed ${SEED} \
           ...
     done
     ```
   * The resulting structure of the codespace is:
     ```text
     data/
     model/
     ├── train_heterogeneous_group_rgcn_gru.py
     ├── run_hetero_group_rgcn_gru_multiseed.sh
     └── runs/
         ├── hetero_group_rgcn_sage_gru/full
             ├── seed123
             ├── seed456
             └── seed123
                 ├── test_metrics.json
                 ├── training_history.csv
                 └── best_hetero_rgcn_gru.pt
         ├── model_2
         ...
     ```
   * The evaluation metrics being used are recall@k (k = 1, 5, 10) and mean/med Haversine distance. By running `summarize_model_runs.py`, one can get summary statistics (including mean and standard deviation) of test metrics for different models across different seeds. The resulting statistics will be stored under `runs/summary`.
   * Subgroup Analyses: Running `eval/eval_dest_poi_group.py`, one can compare test performance for destinations with POIs vs. destinations without POIs.

2. GRU encoded heterogeneous HGT is implemented in `hgt_train.py`, with the model
   defined in `hgt_model.py`. To submit to the Yale cluster, use the provided SLURM
   script `run_hgt_train.sh` (update `DATA_DIR`, `OUT_DIR`, and `LOG_DIR` to match
   your cluster paths before submitting):

   ```bash
   sbatch run_hgt_train.sh

