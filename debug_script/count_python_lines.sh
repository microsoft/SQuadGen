#!/bin/sh
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

python - <<'PY'
from pathlib import Path
import subprocess

root = Path.cwd()
excluded_dirs = {"DualMesh-UDF"}
records = []
total_lines = 0

result = subprocess.run(
    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", "*.py"],
    check=True,
    stdout=subprocess.PIPE,
)

for name in result.stdout.decode("utf-8", errors="replace").split("\0"):
    if not name:
        continue

    relative_path = Path(name)
    if relative_path.parts and relative_path.parts[0] in excluded_dirs:
        continue

    path = root / relative_path
    if not path.is_file():
        continue

    with path.open("r", encoding="utf-8", errors="replace") as file:
        line_count = sum(1 for _ in file)

    total_lines += line_count
    records.append((line_count, relative_path.as_posix()))

records.sort(key=lambda item: (-item[0], item[1]))
line_width = max([len("LINES"), *(len(str(line_count)) for line_count, _ in records)], default=len("LINES"))

print(f"{'LINES':>{line_width}}  FILE")
for line_count, relative_path in records:
    print(f"{line_count:>{line_width}}  {relative_path}")

print("-" * (line_width + 2 + 4))
print(f"{total_lines:>{line_width}}  TOTAL")
PY