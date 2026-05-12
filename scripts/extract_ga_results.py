"""Compatibility wrapper for GA-only aggregate extraction.

Prefer:
    envs/esnas/Scripts/python scripts/print_saved_results.py --methods ga
"""
from print_saved_results import collect_method


if __name__ == '__main__':
    collect_method('ga', write=True)
