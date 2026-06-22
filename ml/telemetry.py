from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
