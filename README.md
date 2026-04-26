# Context-Aware Taxi Destination Prediction via Heterogeneous Urban Graphs
A final project for CPSC5830.

**Group Member:** Charles Cai & Yidan Mei


## Raw data

This project uses the Porto taxi trajectory dataset from the ECML/PKDD 2015 Taxi Trajectory Prediction challenge. The preprocessing script expects the raw Kaggle files to be placed in the data directory as:

```text
data/
├── train.csv
└── test.csv
```

Below is the sample code to download the raw dataset:

```bash
pip install kagglehub
```

```python
import kagglehub
import os
import zipfile
import shutil
from pathlib import Path

# Download Kaggle data.
cache_dir = Path(
    kagglehub.competition_download(
        "pkdd-15-predict-taxi-service-trajectory-i"
    )
)

# Unzip downloaded files.
for f in cache_dir.iterdir():
    if f.suffix == ".zip":
        with zipfile.ZipFile(f, "r") as z:
            z.extractall(cache_dir)

# Copy train.csv and test.csv to the project data directory.
data_dir = Path("data")
data_dir.mkdir(parents=True, exist_ok=True)

for name in ["train.csv", "test.csv"]:
    src = cache_dir / name
    if src.exists():
        shutil.copy2(src, data_dir / name)

print("Files in data/:", os.listdir(data_dir))
```

## Data Preparation

The preprocessing script is located at `data_prep/prepare_data3.py`. It takes the raw Porto taxi files, `train.csv` and optionally `test.csv`, and converts them into the processed data bundle used by our destination prediction task. The script builds prefix-to-destination supervised examples, constructs a heterogeneous urban graph from trajectory transitions, OpenStreetMap roads, and OpenStreetMap POIs, and saves the processed outputs as a compressed `.tar.gz` bundle.

The expected raw data layout is:

After running the preprocessing script, the main output is:

```text
data/trial2/porto_data_bundle_trial2.tar.gz
```

This bundle contains the processed heterogeneous graph, ID mappings, feature names, preprocessing summary, Kaggle test prefixes, and sharded supervised train/validation/test examples. This `.tar.gz` file is the data artifact used by the downstream modeling code.

A SLURM example script is provided in `run_prepare_data3.sh`. It contains the command for running `data_prep/prepare_data3.py`, but users should edit the paths before running it. In particular, update the raw data directory, output directory, log directory, Python environment, and any cluster-specific resource settings. The default paths inside `prepare_data3.py` may also need to be changed or overridden through command-line arguments such as `--data-dir` and `--out-dir`.

Example command:

```bash
python data_prep/prepare_data3.py \
  --data-dir data \
  --out-dir data/trial2 \
  --cell-size 250 \
  --place "Porto, Portugal" \
  --train-frac 0.70 \
  --val-frac 0.15 \
  --num-prefix-samples 5 \
  --chunksize 50000
```
