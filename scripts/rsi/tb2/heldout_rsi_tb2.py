# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Score the seed and final agent-core harnesses on the held-out TB2 tasks.

59 tasks x (seed, final) x 2 trials = 236 Harbor episodes, 4 at a time, through
the same backend the search used. Rows go to a CSV with the reef arm's schema:
arm,harness,task,trial,reward,exception_type,agent_started

Re-running skips rows already in the CSV. Seed and final episodes of a task are
interleaved so both see the same server conditions.

Usage:
    python heldout_rsi_tb2.py --run-dir <search output> --cases cases_heldout.json \
        --port 8081 --out heldout/agentcore_heldout.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import uuid
from pathlib import Path

import yaml
from run_rsi_tb2 import seed_harness_refs

from openjiuwen.rsi.harness_rsi.config import EvaluatorConfig
from openjiuwen.rsi.harness_rsi.evaluator import harbor_runtime
from openjiuwen.rsi.harness_rsi.evaluator.case_backend import SingleHarnessExecutionBackend
from openjiuwen.rsi.harness_rsi.single_harness import load_cases

ARM = "agent-core"
COLUMNS = ["arm", "harness", "task", "trial", "reward", "exception_type", "agent_started"]
TB2_DIR = Path(__file__).resolve().parent


def solver_path(harness_refs_path: str | Path) -> str:
    refs = yaml.safe_load(Path(harness_refs_path).read_text(encoding="utf-8")) or {}
    refs = refs.get("harness_refs", refs)
    paths = [str(path) for path in refs.values() if isinstance(path, str) and path.strip()]
    if len(paths) != 1:
        raise ValueError(f"expected one harness ref in {harness_refs_path}, got {paths}")
    return paths[0]


def harnesses(run_dir: Path, only: str) -> dict[str, str]:
    """Seed = the run's copied seed harness; final = the best harness the search promoted."""
    found: dict[str, str] = {}
    if only in ("both", "seed"):
        found["seed"] = solver_path(seed_harness_refs(run_dir))
    if only in ("both", "final"):
        summary = json.loads((run_dir / "tb2_summary.json").read_text(encoding="utf-8"))
        found["final"] = solver_path(summary["best_harness_refs_path"])
    return found


def done_rows(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.is_file():
        return set()
    with csv_path.open(encoding="utf-8", newline="") as file:
        return {(row["harness"], row["task"], int(row["trial"])) for row in csv.DictReader(file)}


async def main(args: argparse.Namespace) -> int:
    import harbor  # noqa: F401  (fail fast: Harbor needs Python 3.12 and harbor==0.20.0)

    os.environ["RSI_TB2_PORT"] = str(args.port)
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(args.out).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir).expanduser().resolve() if args.work_dir else out_csv.with_suffix("")
    harbor_runtime.configure_harbor_trials(trials_root=work_dir / "harbor_trials", model_name=args.model_name)
    backend = SingleHarnessExecutionBackend(
        config=EvaluatorConfig(model_config_ref=str(Path(args.model_config).expanduser().resolve()))
    )
    selected = harnesses(run_dir, args.only)
    print(f"[heldout] harnesses={selected}", flush=True)
    cases = load_cases([str(Path(args.cases).expanduser().resolve())])
    skip = done_rows(out_csv)
    jobs = [
        (label, path, case, trial)
        for trial in range(1, args.trials + 1)
        for case in cases
        for label, path in selected.items()
        if (label, case["case_id"], trial) not in skip
    ]
    print(f"[heldout] {len(jobs)} episodes to run, {len(skip)} already in {out_csv}", flush=True)
    if not out_csv.is_file():
        with out_csv.open("w", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(COLUMNS)

    slots = asyncio.Semaphore(args.concurrency)
    failures: list[str] = []

    async def run_episode(label: str, harness_path: str, case: dict, trial: int) -> None:
        case_id = case["case_id"]
        async with slots:
            try:
                result = await backend.execute(
                    case=dict(case),
                    output_dir=str(work_dir / "episodes" / f"{label}__{case_id}__t{trial}"),
                    session_id=f"heldout-{label}-{case_id}-t{trial}-{uuid.uuid4().hex[:8]}",
                    harness_refs={"solver": harness_path},
                )
            except Exception as exc:  # noqa: BLE001 - no row is written, so a rerun retries it
                failures.append(f"{label}/{case_id}/t{trial}: {type(exc).__name__}: {exc}")
                print(f"[heldout] FAILED {failures[-1]}", flush=True)
                return
        info = result.metadata["terminal_bench"]
        row = [
            ARM,
            label,
            case_id,
            trial,
            result.judge_result.score,
            info["exception_type"],
            info["agent_started"],
        ]
        with out_csv.open("a", encoding="utf-8", newline="") as file:
            csv.writer(file).writerow(row)
        print(f"[heldout] {label} {case_id} t{trial} reward={row[4]} exception={row[5] or '-'}", flush=True)

    await asyncio.gather(*(run_episode(*job) for job in jobs))
    print(f"[heldout] finished; {len(failures)} episodes failed without a row (rerun to retry)", flush=True)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, help="search run directory (has tb2_summary.json)")
    parser.add_argument("--cases", required=True, help="held-out cases JSON from convert_tb2.py")
    parser.add_argument("--port", type=int, required=True, help="llama-server port")
    parser.add_argument("--out", required=True, help="CSV to append rows to")
    parser.add_argument("--work-dir", default="", help="episode outputs (default: next to the CSV)")
    parser.add_argument("--model-config", default=str(TB2_DIR / "models" / "qwen.yaml"))
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--only", choices=["both", "seed", "final"], default="both")
    parser.add_argument("--model-name", default="qwen27b")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
