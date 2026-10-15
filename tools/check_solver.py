"""Solve a reference target with gpu_solve and compare the digests with tools/expected_digests.json.

Usage: uv run tools/check_solver.py {KQvKR|KQBvKR|all5} OUTPUT_DIRECTORY [PACK_THREADS]

KQvKR and KQBvKR solve one capture tree and compare the payload digests of the
.gpu.tbp files. all5 solves every pawnless table of five pieces or fewer and
compares the content digests of the .tbz files. These do not change with the
zstd level. The output directory must be empty or absent, because gpu_solve
uses the finished files in it again. Build gpu_solve before you use this program.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOLVER = ROOT / "rust/gpu/target/release/gpu_solve"
EXPECTED = json.loads((Path(__file__).parent / "expected_digests.json").read_text())


def main() -> int:
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in EXPECTED:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    target, output = sys.argv[1], Path(sys.argv[2])
    if output.exists() and any(output.iterdir()):
        print(f"{output} is not empty", file=sys.stderr)
        return 2
    if target == "all5":
        threads = sys.argv[3] if len(sys.argv) == 4 else "20"
        command = [SOLVER, "--all", "5", output, "15", threads]
        key = "content_sha256"
    else:
        command = [SOLVER, target, output]
        key = "payload_sha256"
    run = subprocess.run(command, capture_output=True, text=True, check=False)  # noqa: S603 - no shell
    if run.returncode != 0:
        print(run.stderr[-2000:], file=sys.stderr)
        print(f"gpu_solve failed with exit code {run.returncode}")
        return 1
    report = json.loads(run.stdout)
    found = {t["material"]: t[key] for t in report["tables"]}
    expected = EXPECTED[target]
    wrong = sorted(m for m in expected.keys() | found.keys() if found.get(m) != expected.get(m))
    print(
        json.dumps(
            {
                "target": target,
                "tables": len(found),
                "expected_tables": len(expected),
                "different_or_missing": wrong,
                "wall_seconds": round(report["wall_seconds"], 2),
                "status": "fail" if wrong else "pass",
            }
        )
    )
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
