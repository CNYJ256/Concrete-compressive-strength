from __future__ import annotations

"""数据预处理入口：执行基础校验并产出数据摘要。"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from concrete_compressive_strength.core import BASE_FEATURES, TARGET_COL, load_data  # noqa: E402


def main() -> None:
    data_path = ROOT / "data" / "Concrete_Data.xls"
    out_dir = ROOT / "results" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(data_path)

    summary = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "features": BASE_FEATURES,
        "target": TARGET_COL,
        "target_mean": float(np.mean(df[TARGET_COL])),
        "target_std": float(np.std(df[TARGET_COL])),
        "missing_rows": int(df.isna().any(axis=1).sum()),
    }

    output = out_dir / "dataset_profile.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Dataset profile saved to: {output}")


if __name__ == "__main__":
    main()
