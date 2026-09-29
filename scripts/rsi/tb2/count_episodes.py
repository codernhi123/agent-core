# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Count Harbor episodes in a trials directory (shared definition with the reef arm).

Episode = a Harbor trial whose agent started (``agent_execution`` recorded);
a missing reward counts as 0.

Usage: python count_episodes.py <run_dir>/harbor_trials [more dirs...]
"""

import argparse
import collections
import json
from pathlib import Path


def summarize(trials_dirs: list[Path]) -> dict:
    launched = finished = episodes = passed = 0
    reward_sum = 0.0
    exceptions: collections.Counter = collections.Counter()
    for trials_dir in trials_dirs:
        for trial_dir in sorted(path for path in trials_dir.iterdir() if (path / "config.json").is_file()):
            launched += 1
            result_path = trial_dir / "result.json"
            if not result_path.is_file():
                continue  # still running, or killed before Harbor wrote a result
            finished += 1
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("exception_info"):
                exceptions[result["exception_info"].get("exception_type", "?")] += 1
            if result.get("agent_execution") is None:
                continue
            episodes += 1
            reward = float(((result.get("verifier_result") or {}).get("rewards") or {}).get("reward", 0.0) or 0.0)
            reward_sum += reward
            passed += reward >= 1.0
    return {
        "trials_launched": launched,
        "trials_finished": finished,
        "episodes": episodes,
        "passed": passed,
        "pass_rate": round(passed / episodes, 4) if episodes else None,
        "reward_sum": reward_sum,
        "exceptions": dict(exceptions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trials_dirs", nargs="+")
    args = parser.parse_args()
    print(json.dumps(summarize([Path(path) for path in args.trials_dirs]), indent=2))


if __name__ == "__main__":
    main()
