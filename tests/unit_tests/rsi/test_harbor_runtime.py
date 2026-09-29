# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for running single-harness Terminal-Bench cases as Harbor trials."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openjiuwen.rsi.harness_rsi.evaluator.errors import EvaluationInfrastructureError
from openjiuwen.rsi.harness_rsi.evaluator.harbor_runtime import (
    EpisodeBudget,
    HarborShellOperation,
    container_workdir,
    judge_from_trial,
    live_agent,
)
from openjiuwen.rsi.harness_rsi.evaluator.judger import ScriptBasedJudger
from openjiuwen.rsi.harness_rsi.evaluator.terminal_bench_runtime import TerminalBenchCommandRecorder


class _FakeEnvironment:
    def __init__(self, *, return_code: int = 0, stdout: str = "", error: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._return_code = return_code
        self._stdout = stdout
        self._error = error

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append({"command": command, "cwd": cwd, "env": env, "timeout_sec": timeout_sec})
        if self._error is not None:
            raise self._error
        return SimpleNamespace(stdout=self._stdout, stderr="", return_code=self._return_code)


def _operation(tmp_path: Path, environment: _FakeEnvironment) -> HarborShellOperation:
    return HarborShellOperation(
        environment=environment,
        host_workspace_dir=tmp_path,
        container_workspace_dir="/app",
        recorder=TerminalBenchCommandRecorder(),
    )


def _trial(*, rewards=None, started=True, exception_type=""):
    return SimpleNamespace(
        trial_name="extract-elf__abc",
        trial_uri="file:///trials/extract-elf__abc",
        agent_execution=SimpleNamespace() if started else None,
        verifier_result=SimpleNamespace(rewards=rewards) if rewards is not None else None,
        exception_info=(
            SimpleNamespace(exception_type=exception_type, exception_message="boom") if exception_type else None
        ),
    )


@pytest.mark.asyncio
async def test_command_runs_in_container_with_in_container_timeout(tmp_path: Path) -> None:
    environment = _FakeEnvironment(stdout="ok\n")
    operation = _operation(tmp_path, environment)

    result = await operation.execute_cmd("echo 'hi'", cwd=str(tmp_path / "src"), timeout=30)

    call = environment.calls[0]
    assert call["command"] == "timeout --signal=TERM --kill-after=5s 30s bash -c 'echo '\"'\"'hi'\"'\"''"
    assert call["cwd"] == "/app/src"
    assert call["timeout_sec"] == 40
    assert result.data.exit_code == 0
    assert result.data.stdout == "ok\n"
    assert operation.command_log()[0]["command"] == "echo 'hi'"


@pytest.mark.asyncio
async def test_command_timeout_is_reported_to_the_agent_not_raised(tmp_path: Path) -> None:
    killed = await _operation(tmp_path, _FakeEnvironment(return_code=124)).execute_cmd("sleep 99", timeout=5)
    hung = await _operation(
        tmp_path, _FakeEnvironment(error=RuntimeError("Command timed out after 15 seconds"))
    ).execute_cmd("sleep 99", timeout=5)

    assert killed.data.exit_code == 124
    assert "command timed out after 5s" in killed.data.stderr
    assert hung.data.exit_code is None
    assert "command timed out after 5s" in hung.data.stderr


@pytest.mark.asyncio
async def test_other_environment_errors_propagate(tmp_path: Path) -> None:
    operation = _operation(tmp_path, _FakeEnvironment(error=RuntimeError("container is gone")))

    with pytest.raises(RuntimeError, match="container is gone"):
        await operation.execute_cmd("ls")


@pytest.mark.asyncio
async def test_container_workdir_reads_image_workdir() -> None:
    assert await container_workdir(_FakeEnvironment(stdout="/app\n")) == "/app"


def test_passing_trial_scores_reward_and_reports_tests(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    verifier.mkdir()
    (verifier / "test-stdout.txt").write_text("2 passed, 1 failed", encoding="utf-8")
    (verifier / "ctrf.json").write_text(
        json.dumps(
            {
                "results": {
                    "tests": [
                        {"name": "test_a", "status": "passed"},
                        {"name": "test_b", "status": "failed"},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    result = judge_from_trial(_trial(rewards={"reward": 0.0}), tmp_path, case_id="extract-elf")

    assert result.score == 0.0
    assert not result.passed
    assert result.metadata["test_output_excerpt"] == "2 passed, 1 failed"
    tests_status = result.metadata["instance_report"]["extract-elf"]["tests_status"]
    assert tests_status["FAIL_TO_PASS"] == {"success": ["test_a"], "failure": ["test_b"]}
    assert judge_from_trial(_trial(rewards={"reward": 1}), tmp_path, case_id="extract-elf").passed


def test_agent_timeout_keeps_verifier_reward(tmp_path: Path) -> None:
    result = judge_from_trial(
        _trial(rewards={"reward": 1.0}, exception_type="AgentTimeoutError"), tmp_path, case_id="c"
    )

    assert result.passed
    assert result.metadata["exception_type"] == "AgentTimeoutError"


def test_missing_reward_scores_zero(tmp_path: Path) -> None:
    no_verifier = judge_from_trial(_trial(exception_type="RuntimeError"), tmp_path, case_id="c")
    not_started = judge_from_trial(_trial(started=False, exception_type="DockerError"), tmp_path, case_id="c")

    assert no_verifier.score == 0.0 and no_verifier.metadata["agent_started"]
    assert not_started.score == 0.0 and not not_started.metadata["agent_started"]
    assert "before the agent started" in not_started.reason


@pytest.mark.asyncio
async def test_budget_blocks_at_limit_until_cancelled() -> None:
    budget = EpisodeBudget(3, already_started=2)
    await budget.admit()
    budget.finish()

    blocked = asyncio.create_task(budget.admit())
    await asyncio.sleep(0.01)

    assert budget.exhausted
    assert not blocked.done()
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    assert budget.started == 3


def test_unknown_live_token_is_an_infrastructure_error() -> None:
    with pytest.raises(EvaluationInfrastructureError):
        live_agent("missing")


def test_judger_accepts_terminal_bench_case_with_task_dir(tmp_path: Path) -> None:
    judger = ScriptBasedJudger()

    judger.validate_case({"case_id": "c", "input": "x", "terminal_bench": {"task_dir": str(tmp_path)}})
    with pytest.raises(EvaluationInfrastructureError):
        judger.validate_case({"case_id": "c", "input": "x", "terminal_bench": {"task_dir": str(tmp_path / "no")}})


def _fake_harbor(monkeypatch: pytest.MonkeyPatch, run_seconds: float) -> None:
    """Install minimal harbor modules so run_harbor_trial runs without Docker."""
    import sys
    import types

    class _Config:
        @classmethod
        def model_validate(cls, data):
            config = cls()
            config.__dict__.update(data)
            return config

        def model_dump(self):
            return vars(self)

    class _Trial:
        def __init__(self, trials_dir: Path) -> None:
            self.paths = SimpleNamespace(trial_dir=trials_dir / "t1")
            self.result = _trial(started=True, exception_type="CancelledError")

        @classmethod
        async def create(cls, config):
            return cls(Path(config.trials_dir))

        async def run(self):
            await asyncio.sleep(run_seconds)
            return _trial(rewards={"reward": 1.0})

    config_module = types.ModuleType("harbor.models.trial.config")
    config_module.TaskConfig = config_module.TrialConfig = _Config
    trial_module = types.ModuleType("harbor.trial.trial")
    trial_module.Trial = _Trial
    monkeypatch.setitem(sys.modules, "harbor.models.trial.config", config_module)
    monkeypatch.setitem(sys.modules, "harbor.trial.trial", trial_module)


@pytest.mark.asyncio
async def test_episode_over_the_cap_scores_zero_and_frees_the_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from openjiuwen.rsi.harness_rsi.evaluator import harbor_runtime

    _fake_harbor(monkeypatch, run_seconds=5)
    monkeypatch.setattr(harbor_runtime, "EPISODE_TIMEOUT_SECONDS", 0.05)
    budget = EpisodeBudget(10)
    monkeypatch.setattr(harbor_runtime, "SETTINGS", harbor_runtime.HarborTrialSettings(budget=budget))

    async def run_agent(instruction, environment):
        return None

    result, trial_dir = await harbor_runtime.run_harbor_trial(
        task_dir=tmp_path, run_agent=run_agent, trials_dir=tmp_path
    )
    judged = judge_from_trial(result, trial_dir, case_id="c")

    assert judged.score == 0.0
    assert judged.metadata["exception_type"] == harbor_runtime.EPISODE_TIMEOUT_EXCEPTION
    assert (budget.started, budget.finished) == (1, 1)


@pytest.mark.asyncio
async def test_episode_under_the_cap_keeps_its_reward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from openjiuwen.rsi.harness_rsi.evaluator import harbor_runtime

    _fake_harbor(monkeypatch, run_seconds=0)
    monkeypatch.setattr(harbor_runtime, "SETTINGS", harbor_runtime.HarborTrialSettings())

    async def run_agent(instruction, environment):
        return None

    result, trial_dir = await harbor_runtime.run_harbor_trial(
        task_dir=tmp_path, run_agent=run_agent, trials_dir=tmp_path
    )

    assert judge_from_trial(result, trial_dir, case_id="c").passed
