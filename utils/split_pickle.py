# to split the massive dump from VDA
import numpy as np
from pathlib import Path
from utils.freeze import freeze_array
from typing import Any, Dict 
import pickle

def bulk_load_depths(path: Path) -> np.ndarray | None:
        if not path.is_file():
            return None
        return freeze_array(np.load(str(path))["depths"]) 

def bulk_load_sam3(path: Path) -> Dict[str, Any] | None:
        if not path.is_file():
            return None
        with open(path, "rb") as fh:
            raw = pickle.load(fh)
        return raw
      