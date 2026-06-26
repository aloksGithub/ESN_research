"""Scan all pickle-bearing files under results/ using picklescan's engine.

picklescan's CLI (`-p <dir>`) silently skips extensions that aren't in its
hardcoded allowlist (notably .obj, and .npy during directory walks). This
wrapper feeds every relevant file directly to scan_file_path so nothing is
skipped, while still benefiting from the maintained blocklist.
"""

import glob
import sys

from picklescan.scanner import SafetyLevel, scan_file_path

EXTENSIONS = (
    "pkl", "pickle", "obj",
    "pt", "pth", "ckpt", "bin",
    "npy", "npz",
    "joblib", "dat", "data",
    "zip", "7z",
)

# Dangerous-tier globals that are expected for this project's pickle workflow.
# Anything outside this set will fail the scan.
#   functools.partial      - skopt stores BO acquisition functions as partials
#   builtins.getattr       - deap GA pickles use it for method lookup
KNOWN_GLOBALS = {
    ("functools", "partial"),
    ("builtins", "getattr"),
}

# Suspicious-tier filter: globals from these top-level packages are expected
# in this project's pickles (numerical libraries, our own code) and are
# suppressed from the FYI summary. Dangerous-tier classification is unaffected.
TRUSTED_TOP_MODULES = frozenset({
    "numpy", "scipy",
    "reservoirpy", "deap",
    "src",                       # this project's own code
    "typing", "collections", "copy", "_operator",
})

# Specific builtins used as data-reconstruction primitives (not callable code).
# These appear constantly in numpy/torch/sklearn pickles; suppress from FYI.
TRUSTED_BUILTINS = frozenset({
    "int", "float", "str", "bool", "bytes",
    "list", "dict", "tuple", "set", "frozenset",
    "type", "map",
})


def _is_expected_suspicious(g) -> bool:
    if g.module == "builtins":
        return g.name in TRUSTED_BUILTINS
    return g.module.split(".", 1)[0] in TRUSTED_TOP_MODULES

if __name__ == "__main__":
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(f"results/**/*.{ext}", recursive=True))

    if not files:
        print("No pickle-bearing files found in results/")
        sys.exit(1)

    print(f"Scanning {len(files)} files...\n")

    dangerous_files = 0
    errored = 0
    suspicious_counts: dict[str, int] = {}

    for path in sorted(files):
        try:
            result = scan_file_path(path)
        except Exception as e:
            errored += 1
            print(f"ERROR: {path}: {e}\n")
            continue

        dangerous = [g for g in result.globals
                     if g.safety is SafetyLevel.Dangerous
                     and (g.module, g.name) not in KNOWN_GLOBALS]
        suspicious = [g for g in result.globals
                      if g.safety is SafetyLevel.Suspicious]

        for g in suspicious:
            if _is_expected_suspicious(g):
                continue
            key = f"{g.module}.{g.name}"
            suspicious_counts[key] = suspicious_counts.get(key, 0) + 1

        if dangerous or result.scan_err:
            dangerous_files += 1
            print(f"DANGEROUS: {path}")
            if result.scan_err:
                print(f"  scan error: {result.scan_err}")
            for g in dangerous:
                print(f"  {g.module}.{g.name}")
            print()

    print(f"Summary: {len(files)} files scanned, "
          f"{dangerous_files} with dangerous globals, {errored} errored")

    if suspicious_counts:
        print(f"\nUnrecognized suspicious globals (not on project allowlist):")
        for name, count in sorted(suspicious_counts.items(),
                                  key=lambda kv: -kv[1]):
            print(f"  {count:5d}x {name}")

    if dangerous_files or errored:
        print("\nReview the flagged files before loading them.")
        sys.exit(1)
    print("\nNo dangerous globals found.")
