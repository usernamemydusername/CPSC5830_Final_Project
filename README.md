# CPSC5830_Final_Project
Context-Aware Taxi Destination Prediction via Heterogeneous Urban Graphs

## Raw data

This project uses the Porto taxi trajectory dataset from the ECML/PKDD 2015 Taxi Trajectory Prediction challenge. The preprocessing script expects the raw Kaggle files to be placed in the data directory as:

```text
data/
├── train.csv
└── test.csv

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

The data preparation step
