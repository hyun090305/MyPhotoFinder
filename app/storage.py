from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import FaceRecord, GroupRecord, PhotoRecord


class JsonStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write({"photos": {}, "faces": {}, "groups": {}})

    def _read(self) -> dict[str, Any]:
        with self.db_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, payload: dict[str, Any]) -> None:
        tmp = self.db_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(self.db_path)

    def transaction(self, fn):
        with self._lock:
            payload = self._read()
            result = fn(payload)
            self._write(payload)
            return result

    def list_groups(self) -> list[GroupRecord]:
        data = self._read()["groups"]
        return [GroupRecord(**g) for g in data.values()]

    def get_snapshot(self) -> dict[str, Any]:
        return self._read()

    def clear(self) -> None:
        with self._lock:
            self._write({"photos": {}, "faces": {}, "groups": {}})
