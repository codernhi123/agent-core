# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Turn Terminal-Bench 2 task folders into agent-core's case file.

Usage:
    python convert_tb2.py --tasks-root /path/to/terminal-bench-2 \
        --task-list /path/to/search_tasks_20.txt --out cases_search20.json

Each case carries the task's instruction.md as its input and the absolute task
directory, which the ``terminal_bench`` backend hands to Harbor. Case order
follows the task list. A missing task stops the conversion.
"""

import argparse
import json
import tomllib
from pathlib import Path


def convert(tasks_root: Path, task_names: list[str]) -> dict:
    cases = []
    for name in task_names:
        task_dir = (tasks_root / name).resolve()
        instruction_path = task_dir / "instruction.md"
        toml_path = task_dir / "task.toml"
        if not instruction_path.is_file() or not toml_path.is_file():
            raise FileNotFoundError(f"task {name!r} is missing instruction.md or task.toml under {task_dir}")
        metadata = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("metadata", {})
        cases.append(
            {
                "case_id": name,
                "input": instruction_path.read_text(encoding="utf-8"),
                "difficulty": str(metadata.get("difficulty", "unknown")),
                "dimension": str(metadata.get("category", "unknown")),
                "source": "terminal_bench",
                "task_type": "terminal",
                "terminal_bench": {"task_dir": str(task_dir)},
            }
        )
    return {"cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description="Turn TB2 task folders into agent-core's case file")
    parser.add_argument("--tasks-root", "--task-root", required=True, help="Terminal-Bench 2 checkout")
    parser.add_argument("--task-list", required=True, help="text file, one task name per line")
    parser.add_argument("--out", "--output", required=True, help="cases JSON to write")
    args = parser.parse_args()

    tasks_root = Path(args.tasks_root).expanduser()
    if not tasks_root.is_dir():
        raise NotADirectoryError(f"tasks root does not exist: {tasks_root}")
    lines = Path(args.task_list).expanduser().read_text(encoding="utf-8").splitlines()
    task_names = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    if len(set(task_names)) != len(task_names):
        raise ValueError(f"duplicate task names in {args.task_list}")

    payload = convert(tasks_root, task_names)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['cases'])} cases to {out}")


if __name__ == "__main__":
    main()
