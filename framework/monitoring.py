from __future__ import annotations
import time
from datetime import datetime
from pathlib import Path
import pandas as pd


class StepMonitor:
    def __init__(self, out_dir: Path) -> None:
        self._path = out_dir / "execution_log.csv"
        self._log: list[dict] = []
        self._t0 = time.time()

    def step(self, name: str, details: str = "") -> None:
        elapsed = round(time.time() - self._t0, 2)
        self._log.append({
            "step": name,
            "details": details,
            "elapsed_s": elapsed,
            "timestamp": datetime.now().isoformat(),
        })
        pd.DataFrame(self._log).to_csv(self._path, index=False)
        suffix = f" — {details}" if details else ""
        print(f"  [{elapsed:8.1f}s] {name}{suffix}")

    def save(self) -> pd.DataFrame:
        df = pd.DataFrame(self._log)
        df.to_csv(self._path, index=False)
        return df
