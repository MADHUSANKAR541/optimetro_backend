import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent / "sample_data"

def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(BASE_DIR / name)

def load_json(name: str):
    with open(BASE_DIR / name, 'r') as f:
        return json.load(f)
