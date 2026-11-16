"""PRISM platform configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "db" / "prism.db"
UPLOADS_DIR = DATA_DIR / "uploads"
RUNS_DIR = DATA_DIR / "runs"
REPORTS_DIR = DATA_DIR / "reports"
SIMULATION_DIR = BASE_DIR / "simulation"

for d in (DATA_DIR / "db", UPLOADS_DIR, RUNS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

DATABASE_URL = f"sqlite:///{DB_PATH}"
API_PREFIX = "/api/v1"
PYTHON_EXE = "D:/ruangong/conda-envs/robocasa/python.exe"
