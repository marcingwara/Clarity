import json
import os
from pathlib import Path
from typing import Any, Dict, List

# Better for Streamlit Cloud than "next to repo"
DEFAULT_PATH = Path(os.path.expanduser("~")) / "decisions.jsonl"


def load_decisions(path: Path = DEFAULT_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                # ignore corrupted lines (never crash the app)
                continue
    return items


def append_decision(record: Dict[str, Any], path: Path = DEFAULT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    return path


def delete_decision_by_timestamp(timestamp: str, path: Path = DEFAULT_PATH) -> bool:
    """Deletes the first decision with matching timestamp. Returns True if deleted."""
    if not path.exists():
        return False

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    before = len(rows)
    rows = [r for r in rows if r.get("timestamp") != timestamp]
    after = len(rows)

    if after == before:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return True


def rewrite_all_decisions(decisions: List[Dict[str, Any]], path: Path = DEFAULT_PATH) -> Path:
    """Rewrite the whole decisions.jsonl file from a list of decision dicts."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return path