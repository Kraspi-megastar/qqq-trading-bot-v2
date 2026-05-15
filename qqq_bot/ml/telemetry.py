from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return str(value)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
