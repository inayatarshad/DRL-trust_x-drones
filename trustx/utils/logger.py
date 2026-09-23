"""Minimal CSV + console logger for training runs."""

from __future__ import annotations

import csv
import time
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: list[str], echo_every: int = 10):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._echo_every = echo_every
        self._count = 0
        self._t0 = time.time()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self._count += 1
        if self._echo_every and self._count % self._echo_every == 0:
            elapsed = time.time() - self._t0
            fields = "  ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()
            )
            print(f"[{elapsed:7.1f}s] {fields}", flush=True)

    def close(self) -> None:
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
