import json
from pathlib import Path
from typing import Any, Dict, List

# Always save next to this file's directory
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = PROJECT_DIR / "decisions.jsonl"


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

    rows: List[Dict[str,Any]]= []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    before = len(rows)
    rows = [r for r in rows if r.get("timestamp") != timestamp]
    after = len(rows)

    if after == before:
        return False

    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return True

def rewrite_all_decisions(decisions:List[Dict[str,Any]], path: Path = DEFAULT_PATH):
    """Rewrite the whole decisions.json file from a list of decision dicts."""
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    return path