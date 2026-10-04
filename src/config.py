from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
LEAD_TIME = 7  
QUANTILES = [0.10, 0.50, 0.90]
RANDOM_STATE = 42
UNIT_STOCKOUT_COST = 12.00  
UNIT_HOLDING_COST = 1.33