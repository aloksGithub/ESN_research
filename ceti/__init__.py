"""CeTI-Locomotion data preparation and sequence classification helpers."""

from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "CeTI-Locomotion"
DEFAULT_SPLIT = Path(__file__).resolve().parents[1] / "results/ceti_timing/subject_split.json"
