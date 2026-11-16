"""Worker entry point. A run directory must contain a validated config.json."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.system.config import ExperimentConfig
from experiments.system.pipeline import run_experiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = ExperimentConfig.from_dict(json.loads((args.run_dir / "config.json").read_text(encoding="utf-8-sig")))
    run_experiment(config, args.run_dir)
