from __future__ import annotations
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class OutputDirs:
    base:         Path
    metrics:      Path
    predictions:  Path
    plots:        Path
    tables:       Path
    interpretations: Path
    monitoring:   Path
    summaries:    Path
    eda:          Path
    qualitative:  Path

    def all_dirs(self) -> list[Path]:
        return [self.base, self.metrics, self.predictions, self.plots,
                self.tables, self.interpretations, self.monitoring,
                self.summaries, self.eda, self.qualitative]

    def mkdir_all(self) -> None:
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)


def create_run_dirs(
    model_slug: str,
    dataset_slug: str,
    debug_mode: bool,
    in_colab: bool,
    base_root: Path | None = None,
    script_path: Path | None = None,
) -> OutputDirs:
    ts   = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    slug = f"{ts}_{model_slug}_{dataset_slug}" + ("_debug" if debug_mode else "")

    if base_root is not None:
        root = base_root / slug
    elif in_colab:
        root = Path("/content/outputs") / slug
    else:
        root = (script_path or Path(".")).parent.parent / "outputs" / slug

    dirs = OutputDirs(
        base           = root,
        metrics        = root / "metrics",
        predictions    = root / "predictions",
        plots          = root / "plots",
        tables         = root / "tables",
        interpretations= root / "interpretations",
        monitoring     = root / "monitoring",
        summaries      = root / "summaries",
        eda            = root / "plots" / "eda",
        qualitative    = root / "predictions" / "qualitative",
    )
    dirs.mkdir_all()
    return dirs


def save_config(dirs: OutputDirs, config: dict[str, Any]) -> None:
    p = dirs.base / "config.json"
    with open(p, "w") as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Config saved: {p}")


def save_df(df, filename: str, out_dir: Path) -> Path:
    import pandas as pd
    df = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    p  = out_dir / filename
    df.to_csv(p, index=False)
    print(f"  Saved: {p}")
    return p


def zip_run_outputs(dirs: OutputDirs) -> Path:
    zip_path = shutil.make_archive(str(dirs.base), "zip", root_dir=str(dirs.base))
    print(f"Archive: {zip_path}")
    return Path(zip_path)


def drive_backup(dirs: OutputDirs, drive_available: bool) -> None:
    if not drive_available:
        print(f"Local results: {dirs.base}")
        return
    dr = Path("/content/drive/MyDrive/LinguoMT-AfricaS2T") / dirs.base.name
    if dr.exists():
        shutil.rmtree(str(dr))
    shutil.copytree(str(dirs.base), str(dr))
    zip_path = shutil.make_archive(str(dr), "zip", root_dir=str(dirs.base))
    print(f"Drive backup: {dr}")
    print(f"ZIP:          {zip_path}")


def colab_download(dirs: OutputDirs) -> None:
    try:
        from google.colab import files
        zip_p = dirs.base.parent / f"{dirs.base.name}.zip"
        if zip_p.exists():
            files.download(str(zip_p))
    except Exception:
        pass
