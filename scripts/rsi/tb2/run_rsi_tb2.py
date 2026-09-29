# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Run single-harness RSI on Terminal-Bench 2 through Harbor (agent-core arm).

The equivalent of reef's ``run.py``: it connects the unmodified
``SingleHarnessIterativeOptimizationOrchestrator`` to the model (via the RSI
config) and to Harbor trials (via the ``terminal_bench`` case backend).

Benchmark-matching changes, all made here rather than in the library:
- edit surface limited to prompt + skill (tool/rail edits are host Python);
- every evaluation runs up to ``--concurrency`` cases at once (reef runs 4);
- a hard cap on Harbor trial launches (``--episode-limit``); at the cap the
  run is cancelled, which leaves a resumable ``terminated`` state, and the
  final harness is the best one promoted so far.

Re-running the same command resumes from ``--output``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import urllib.request
from pathlib import Path

import yaml

from openjiuwen.rsi.harness_rsi.config import AutoCoordinatingHarnessConfig
from openjiuwen.rsi.harness_rsi.evaluator import harbor_runtime
from openjiuwen.rsi.harness_rsi.single_harness import iterative
from openjiuwen.rsi.harness_rsi.single_harness.iterative import (
    IterativeSingleHarnessRequest,
    SingleHarnessIterativeOptimizationOrchestrator,
)

EDIT_SURFACE = ["prompt", "skill"]
BUDGET_POLL_SECONDS = 5.0
TB2_DIR = Path(__file__).resolve().parent


class TB2Orchestrator(SingleHarnessIterativeOptimizationOrchestrator):
    """Run batch, gate, and residual evaluations concurrently, like the full evaluation."""

    concurrency = 4

    async def _evaluate(self, *, case_concurrency: int = 1, **kwargs) -> str:
        return await super()._evaluate(case_concurrency=max(case_concurrency, self.concurrency), **kwargs)


def launched_trials(trials_root: Path) -> int:
    """Trials already launched in this run directory (counts toward the cap on resume)."""
    if not trials_root.is_dir():
        return 0
    return sum(1 for path in trials_root.iterdir() if (path / "config.json").is_file())


def load_config(path: Path) -> AutoCoordinatingHarnessConfig:
    """Load the RSI yaml, resolving relative model refs against its directory."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    refs = data.get("model_configs") or {}
    for role, ref in refs.items():
        if ref and not Path(ref).is_absolute():
            refs[role] = str((path.parent / ref).resolve())
    return AutoCoordinatingHarnessConfig.from_dict(data)


def seed_harness_refs(output_dir: Path) -> Path:
    """Copy the empty seed harness into the run directory and point the refs at it."""
    seed = output_dir / "seed_harness"
    if not seed.is_dir():
        shutil.copytree(TB2_DIR / "seed_harness", seed)
    refs = output_dir / "harness_refs.yaml"
    if not refs.is_file():
        refs.write_text(yaml.safe_dump({"harness_refs": {"solver": str(seed)}}), encoding="utf-8")
    return refs


def check_model_server(port: int) -> None:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as response:
        models = [item.get("id") for item in json.load(response).get("data", [])]
    print(f"[tb2] model server :{port} serves {models}", flush=True)


async def stop_when_budget_spent(run_task: asyncio.Task, budget: harbor_runtime.EpisodeBudget) -> None:
    while not run_task.done():
        if budget.exhausted:
            print(f"[tb2] episode cap reached ({budget.started}/{budget.limit}); stopping the run", flush=True)
            run_task.cancel()
            return
        await asyncio.sleep(BUDGET_POLL_SECONDS)


async def main(args: argparse.Namespace) -> int:
    import harbor  # noqa: F401  (fail fast: Harbor needs Python 3.12 and harbor==0.20.0)

    os.environ["RSI_TB2_PORT"] = str(args.port)
    check_model_server(args.port)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    harness_refs = (
        Path(args.harness_refs).expanduser().resolve() if args.harness_refs else seed_harness_refs(output_dir)
    )
    trials_root = output_dir / "harbor_trials"
    already = launched_trials(trials_root)
    budget = harbor_runtime.EpisodeBudget(args.episode_limit, already_started=already)
    harbor_runtime.configure_harbor_trials(trials_root=trials_root, model_name=args.model_name, budget=budget)
    print(f"[tb2] trials={trials_root} launched_so_far={already} cap={args.episode_limit}", flush=True)
    if budget.exhausted:
        print("[tb2] cap already reached; nothing to run", flush=True)
    else:
        # The orchestrator reads this constant when it is constructed.
        iterative._ALLOWED_ACTION_GROUPS = list(EDIT_SURFACE)
        TB2Orchestrator.concurrency = args.concurrency
        config = load_config(Path(args.config).expanduser().resolve())
        orchestrator = TB2Orchestrator(config)
        request = IterativeSingleHarnessRequest(
            dataset_files=[str(Path(args.cases).expanduser().resolve())],
            harness_refs_path=str(harness_refs),
            output_dir=str(output_dir),
            dataset_id="terminal_bench_2_search",
            resume=args.resume,
            auto_full_baseline=True,
            max_iteration=args.max_epochs,
        )
        run_task = asyncio.create_task(orchestrator.run(request))
        watcher = asyncio.create_task(stop_when_budget_spent(run_task, budget))
        try:
            await run_task
        except asyncio.CancelledError:
            if not budget.exhausted:
                raise
        finally:
            watcher.cancel()

    state_path = output_dir / "single_harness_state.yaml"
    state = yaml.safe_load(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    summary = {
        "status": state.get("status", ""),
        "best_harness_refs_path": state.get("best_harness_refs_path", ""),
        "trials_launched": budget.started,
        "episode_limit": budget.limit,
        "trials_root": str(trials_root),
    }
    (output_dir / "tb2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(TB2_DIR / "rsi_tb2.yaml"), help="RSI yaml")
    parser.add_argument("--cases", required=True, help="cases JSON from convert_tb2.py")
    parser.add_argument("--harness-refs", default="", help="default: copy seed_harness/ into --output")
    parser.add_argument("--port", type=int, required=True, help="llama-server port: 8082 (GPU 7) or 8081 (GPU 6)")
    parser.add_argument("--output", required=True, help="run directory (re-use it to resume)")
    parser.add_argument("--episode-limit", type=int, default=492)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=None, help="default: max_epochs from the config")
    parser.add_argument("--model-name", default="qwen27b", help="recorded in Harbor's agent info")
    parser.add_argument("--resume", action="store_true", help="continue an existing run directory")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
