# coding: utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Run single-harness Terminal-Bench cases as Harbor trials.

Harbor owns the task container, the agent timeout, and the verifier, exactly as
for any other Harbor agent. The DeepAgent's bash tool is routed through
``environment.exec`` of the live trial, and the verifier reward becomes the
case ``JudgeResult``.

Harbor is imported lazily: it needs Python 3.12 and is only installed for
Terminal-Bench runs.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from openjiuwen.core.common.exception.codes import StatusCode
from openjiuwen.core.common.logging import logger
from openjiuwen.core.sys_operation.result import (
    ExecuteCmdBackgroundData,
    ExecuteCmdBackgroundResult,
    ExecuteCmdData,
    ExecuteCmdResult,
)
from openjiuwen.rsi.harness_rsi.evaluator.errors import EvaluationInfrastructureError
from openjiuwen.rsi.harness_rsi.evaluator.judger import JudgeResult
from openjiuwen.rsi.harness_rsi.evaluator.swebench_runtime import _bounded_test_output_excerpt
from openjiuwen.rsi.harness_rsi.evaluator.terminal_bench_runtime import (
    TerminalBenchCommandRecorder,
    TerminalBenchDockerShellOperation,
    TerminalBenchDockerSysOperation,
)

HARBOR_AGENT_IMPORT_PATH = "openjiuwen.rsi.harness_rsi.evaluator.harbor_agent:OpenJiuwenHarborAgent"
JUDGE_METHOD = "terminal_bench_harbor"
REWARD_KEY = "reward"
# Abort the run instead of scoring a dead Docker daemon as a string of zeros.
MAX_CONSECUTIVE_SETUP_FAILURES = 3
# Wall-clock cap per episode, including setup and verification: the reef arm's
# executor timeout. A capped episode has no verifier reward and scores 0.
EPISODE_TIMEOUT_SECONDS = 9000
EPISODE_TIMEOUT_EXCEPTION = "EpisodeTimeoutError"

RunAgent = Callable[[str, Any], Awaitable[None]]

# Harbor builds the agent from JSON-serializable kwargs, so live objects are
# handed over through a token instead of the trial config.
_LIVE_AGENTS: dict[str, RunAgent] = {}


class EpisodeBudget:
    """Cap on Harbor trial launches for one optimization run.

    At the cap, ``admit`` blocks instead of raising: the driver cancels the
    orchestrator once every admitted trial has finished, which persists a
    resumable ``terminated`` state.
    """

    def __init__(self, limit: int, *, already_started: int = 0) -> None:
        self.limit = int(limit)
        self.started = int(already_started)
        self.finished = int(already_started)

    @property
    def exhausted(self) -> bool:
        return self.started >= self.limit and self.finished >= self.started

    async def admit(self) -> None:
        if self.started >= self.limit:
            await asyncio.Event().wait()
        self.started += 1

    def finish(self) -> None:
        self.finished += 1


@dataclass(slots=True)
class HarborTrialSettings:
    """Process-wide settings the driver sets before the orchestrator runs."""

    trials_root: str = ""
    model_name: str = "qwen27b"
    budget: EpisodeBudget | None = None
    consecutive_setup_failures: int = 0


SETTINGS = HarborTrialSettings()


def configure_harbor_trials(
    *,
    trials_root: str | Path,
    model_name: str,
    budget: EpisodeBudget | None = None,
) -> HarborTrialSettings:
    """Point every Terminal-Bench case at one trials directory and budget."""
    SETTINGS.trials_root = str(Path(trials_root).expanduser().resolve())
    SETTINGS.model_name = model_name
    SETTINGS.budget = budget
    SETTINGS.consecutive_setup_failures = 0
    return SETTINGS


def live_agent(token: str) -> RunAgent:
    run_agent = _LIVE_AGENTS.get(token)
    if run_agent is None:
        raise EvaluationInfrastructureError(f"no live agent registered for Harbor token {token!r}")
    return run_agent


class HarborShellOperation(TerminalBenchDockerShellOperation):
    """Run agent shell commands through a live Harbor environment."""

    def __init__(
        self,
        *,
        environment: Any,
        host_workspace_dir: Path,
        container_workspace_dir: str,
        recorder: TerminalBenchCommandRecorder | None = None,
    ) -> None:
        super().__init__(
            container_name="harbor-main",
            host_workspace_dir=host_workspace_dir,
            container_workspace_dir=container_workspace_dir,
            recorder=recorder,
            enforce_in_container_timeout=True,
        )
        self._environment = environment

    async def execute_cmd(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout: int | None = 300,
        environment: dict[str, str] | None = None,
        options: dict[str, Any] | None = None,
        shell_type: Literal["auto", "cmd", "powershell", "bash", "sh"] = "auto",
    ) -> ExecuteCmdResult:
        container_cwd = self._container_cwd(cwd)
        wrapped = f"bash -c {shlex.quote(command)}"
        host_timeout = None
        if timeout is not None:
            # The in-container timeout reaps the command tree; the host timeout
            # only guards against a hung docker client.
            wrapped = f"timeout --signal=TERM --kill-after=5s {max(int(timeout), 1)}s {wrapped}"
            host_timeout = int(timeout) + 10
        try:
            result = await self._environment.exec(
                wrapped,
                cwd=container_cwd,
                env=environment or None,
                timeout_sec=host_timeout,
            )
            exit_code: int | None = result.return_code
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            if exit_code == 124 and timeout is not None:
                stderr += f"\ncommand timed out after {timeout}s"
        except RuntimeError as exc:
            if "timed out" not in str(exc):
                raise
            exit_code, stdout, stderr = None, "", f"command timed out after {timeout}s"
        if self._recorder is not None:
            self._recorder.record(
                command=command,
                cwd=container_cwd,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timeout_sec=timeout,
            )
        return ExecuteCmdResult(
            code=StatusCode.SUCCESS.code,
            message=StatusCode.SUCCESS.errmsg,
            data=ExecuteCmdData(
                command=command,
                cwd=container_cwd,
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
            ),
        )

    async def execute_cmd_background(
        self,
        command: str,
        *,
        cwd: str | None = None,
        environment: dict[str, str] | None = None,
        grace: float = 3.0,
        shell_type: Literal["auto", "cmd", "powershell", "bash", "sh"] = "auto",
    ) -> ExecuteCmdBackgroundResult:
        container_cwd = self._container_cwd(cwd)
        wrapped = f"nohup bash -c {shlex.quote(command)} >/dev/null 2>&1 &"
        result = await self._environment.exec(
            wrapped,
            cwd=container_cwd,
            env=environment or None,
            timeout_sec=int(max(grace, 1.0)) + 10,
        )
        if self._recorder is not None:
            self._recorder.record(
                command=command,
                cwd=container_cwd,
                exit_code=result.return_code,
                stdout=result.stdout or "",
                stderr=result.stderr or "",
                timeout_sec=int(max(grace, 1.0)),
                background=True,
            )
        return ExecuteCmdBackgroundResult(
            code=StatusCode.SUCCESS.code,
            message=StatusCode.SUCCESS.errmsg,
            data=ExecuteCmdBackgroundData(command=command, cwd=container_cwd, pid=None),
        )


def build_harbor_sys_operation(
    *,
    sys_operation_id: str,
    environment: Any,
    workspace_dir: Path,
    container_workspace_dir: str,
    recorder: TerminalBenchCommandRecorder | None = None,
) -> TerminalBenchDockerSysOperation:
    return TerminalBenchDockerSysOperation(
        sys_operation_id=sys_operation_id,
        shell_operation=HarborShellOperation(
            environment=environment,
            host_workspace_dir=workspace_dir,
            container_workspace_dir=container_workspace_dir,
            recorder=recorder,
        ),
    )


async def container_workdir(environment: Any) -> str:
    """Return the task container's working directory (the image WORKDIR)."""
    result = await environment.exec("pwd", timeout_sec=60)
    workdir = (result.stdout or "").strip().splitlines()
    return workdir[-1] if result.return_code == 0 and workdir else "/"


async def run_harbor_trial(*, task_dir: str | Path, run_agent: RunAgent, trials_dir: str | Path) -> tuple[Any, Path]:
    """Run one Harbor trial whose agent calls ``run_agent(instruction, environment)``.

    The trial config matches reef-eval's HarborExecutor: task path, trials
    directory, agent, and a docker environment; task timeouts are unchanged.
    """
    from harbor.models.trial.config import TaskConfig, TrialConfig
    from harbor.trial.trial import Trial

    budget = SETTINGS.budget
    if budget is not None:
        await budget.admit()
    token = uuid.uuid4().hex
    _LIVE_AGENTS[token] = run_agent
    try:
        config = TrialConfig.model_validate(
            {
                "task": TaskConfig.model_validate({"path": Path(task_dir)}).model_dump(),
                "trials_dir": Path(trials_dir),
                "agent": {
                    "import_path": HARBOR_AGENT_IMPORT_PATH,
                    "model_name": SETTINGS.model_name,
                    "kwargs": {"live_token": token},
                },
                "environment": {"type": "docker"},
            }
        )
        trial = await Trial.create(config)
        try:
            result = await asyncio.wait_for(trial.run(), timeout=EPISODE_TIMEOUT_SECONDS)
        except TimeoutError:
            # Harbor stops the environment and writes result.json on cancellation.
            result = trial.result
            if result.exception_info is not None:
                result.exception_info.exception_type = EPISODE_TIMEOUT_EXCEPTION
    finally:
        _LIVE_AGENTS.pop(token, None)
        if budget is not None:
            budget.finish()
    if result.agent_execution is None:
        SETTINGS.consecutive_setup_failures += 1
        if SETTINGS.consecutive_setup_failures >= MAX_CONSECUTIVE_SETUP_FAILURES:
            raise EvaluationInfrastructureError(
                f"{SETTINGS.consecutive_setup_failures} consecutive Harbor trials failed before the agent started; "
                f"last error: {_exception_message(result)}"
            )
    else:
        SETTINGS.consecutive_setup_failures = 0
    return result, Path(trial.paths.trial_dir)


def judge_from_trial(result: Any, trial_dir: str | Path, *, case_id: str) -> JudgeResult:
    """Translate a Harbor trial into the evaluator's verifier contract.

    A missing reward scores 0, the same policy as the reef arm. Per-test
    outcomes use the SWE-bench ``instance_report`` shape so the analyzer sees
    which tests failed and the verifier output.
    """
    rewards = dict(result.verifier_result.rewards or {}) if result.verifier_result is not None else {}
    reward = float(rewards.get(REWARD_KEY, 0.0) or 0.0)
    passed = reward >= 1.0
    verifier_dir = Path(trial_dir) / "verifier"
    test_output_path = verifier_dir / "test-stdout.txt"
    tests = _ctrf_tests(verifier_dir / "ctrf.json")
    agent_started = result.agent_execution is not None
    if not agent_started:
        reason = f"Harbor trial failed before the agent started: {_exception_message(result)}"
    elif not rewards:
        reason = f"verifier produced no reward: {_exception_message(result)}"
    else:
        reason = f"verifier reward {reward}"
    return JudgeResult(
        method=JUDGE_METHOD,
        score=reward,
        passed=passed,
        reason=reason,
        metadata={
            "rewards": rewards,
            "agent_started": agent_started,
            "exception_type": result.exception_info.exception_type if result.exception_info is not None else "",
            "trial_uri": str(result.trial_uri),
            "test_output_path": str(test_output_path),
            "test_output_excerpt": _bounded_test_output_excerpt(_read_text(test_output_path)),
            "instance_report": {
                case_id: {
                    "resolved": passed,
                    "tests_status": {
                        "FAIL_TO_PASS": {
                            "success": [name for name, status in tests if status == "passed"],
                            "failure": [name for name, status in tests if status != "passed"],
                        },
                        "PASS_TO_PASS": {"success": [], "failure": []},
                    },
                }
            },
        },
    )


def _ctrf_tests(path: Path) -> list[tuple[str, str]]:
    """Read (name, status) pairs from the pytest CTRF report TB2 verifiers write."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    tests = (payload.get("results") or {}).get("tests") if isinstance(payload, dict) else None
    if not isinstance(tests, list):
        return []
    return [
        (str(test.get("name")), str(test.get("status") or ""))
        for test in tests
        if isinstance(test, dict) and test.get("name")
    ]


def _exception_message(result: Any) -> str:
    info = result.exception_info
    return f"{info.exception_type}: {info.exception_message}"[:2000] if info is not None else ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def log_trial(case_id: str, result: Any) -> None:
    logger.info(
        "[harbor] case={} trial={} agent_started={} rewards={} exception={}",
        case_id,
        result.trial_name,
        result.agent_execution is not None,
        (result.verifier_result.rewards if result.verifier_result is not None else None),
        (result.exception_info.exception_type if result.exception_info is not None else ""),
    )


__all__ = [
    "EpisodeBudget",
    "HarborShellOperation",
    "HarborTrialSettings",
    "SETTINGS",
    "build_harbor_sys_operation",
    "configure_harbor_trials",
    "container_workdir",
    "judge_from_trial",
    "live_agent",
    "log_trial",
    "run_harbor_trial",
]
