"""GEPA adapter for the Harbor pipeline harness.

Bridges gepa.optimize() and the multi-agent pipeline in benchmark/pipeline.py.
A GEPA candidate is a mapping stage_id -> system_prompt. For each evaluation
the adapter writes the candidate's prompts into a copy of pipeline_spec.yaml,
runs `harbor run` on a batch of tasks with PIPELINE_SPEC_PATH pointing at that
copy, and scores each task by the verifier reward (passed = 1.0).

For reflection, the adapter assembles per-stage feedback from:
  - the trial outcome (reward, verifier test output on failure),
  - the stage's input/output from stage_traces.json,
  - the LLM judge in benchmark/evaluator.py (score / missing / assessment).

Division of labor (see program_pipeline.md): GEPA owns prompt *text*; the
meta-agent owns topology, tools, turn budgets, and handoffs. The candidate
never changes pipeline structure — only the system_prompt of existing stages.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gepa.core.adapter import EvaluationBatch

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_SPEC_PATH = Path(__file__).parent / "pipeline_spec.yaml"

# The 10-task triage set from program_pipeline.md.
TRIAGE_TASKS = [
    "modernize-scientific-stack",
    "openssl-selfsigned-cert",
    "prove-plus-comm",
    "nginx-request-logging",
    "configure-git-webserver",
    "cancel-async-tasks",
    "crack-7z-hash",
    "extract-elf",
    "kv-store-grpc",
    "log-summary-date-ranges",
]

MAX_FEEDBACK_CHARS = 3_000
MAX_IO_CHARS = 4_000


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n[... truncated ...]\n" + text[-limit // 2 :]


def _harbor_bin() -> str:
    local = Path(sys.executable).parent / "harbor"
    if local.exists():
        return str(local)
    found = shutil.which("harbor")
    if found:
        return found
    raise RuntimeError("harbor CLI not found; run `uv sync` first.")


@dataclass
class TaskTrajectory:
    """Everything reflection needs about one task's rollout."""

    task: str
    score: float
    instruction: str = ""
    stage_traces: list[dict] = field(default_factory=list)
    test_output: str = ""
    error: str = ""
    judge: dict[str, dict] | None = None  # stage_id -> {score, missing, assessment}


class HarborPipelineAdapter:
    """GEPAAdapter implementation over `harbor run` + benchmark.pipeline:AutoAgent."""

    # Use GEPA's default LLM-based instruction proposer (the engine checks
    # for this attribute explicitly).
    propose_new_texts = None

    def __init__(
        self,
        base_spec_path: Path = BASE_SPEC_PATH,
        dataset: str = "terminal-bench@2.0",
        output_dir: Path | None = None,
        n_concurrent: int = 10,
        n_attempts: int = 1,
        env_file: Path | None = None,
        use_judge: bool = True,
        timeout_sec: int = 3600,
    ):
        self.base_spec = yaml.safe_load(Path(base_spec_path).read_text())
        self.dataset = dataset
        self.output_dir = Path(output_dir) if output_dir else REPO_ROOT / "jobs" / "gepa"
        self.n_concurrent = n_concurrent
        self.n_attempts = n_attempts
        default_env = REPO_ROOT / ".env"
        self.env_file = env_file if env_file else (default_env if default_env.exists() else None)
        self.use_judge = use_judge
        self.timeout_sec = timeout_sec
        self._eval_counter = 0

    # ------------------------------------------------------------- evaluate

    def render_spec(self, candidate: dict[str, str]) -> dict:
        """Base spec with the candidate's prompts substituted in.

        Structure (stages, tools, turns, handoffs) always comes from the base
        spec; only system_prompt text is candidate-controlled.
        """
        spec = deepcopy(self.base_spec)
        for stage in spec["pipeline"]["stages"]:
            if stage["id"] in candidate:
                stage["system_prompt"] = candidate[stage["id"]]
        return spec

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        run_dir = self.output_dir / f"eval-{self._eval_counter:03d}-{uuid.uuid4().hex[:6]}"
        self._eval_counter += 1
        run_dir.mkdir(parents=True, exist_ok=True)

        spec_path = run_dir / "pipeline_spec.yaml"
        spec_path.write_text(yaml.safe_dump(self.render_spec(candidate), sort_keys=False))
        (run_dir / "candidate.json").write_text(json.dumps(candidate, indent=2))

        cmd = [
            _harbor_bin(), "run",
            "--dataset", self.dataset,
            "-a", "benchmark.pipeline:AutoAgent",
            "-o", str(run_dir),
            "--job-name", "trials",
            "-n", str(min(self.n_concurrent, len(batch) * self.n_attempts)),
            "-k", str(self.n_attempts),
            "-q", "-y",
        ]
        for task in batch:
            cmd += ["-i", task]
        if self.env_file:
            cmd += ["--env-file", str(self.env_file)]

        env = {**os.environ, "PIPELINE_SPEC_PATH": str(spec_path)}
        log_path = run_dir / "harbor.log"
        harbor_error = ""
        with open(log_path, "w") as log:
            try:
                subprocess.run(
                    cmd, cwd=REPO_ROOT, env=env,
                    stdout=log, stderr=subprocess.STDOUT,
                    timeout=self.timeout_sec, check=False,
                )
            except subprocess.TimeoutExpired:
                harbor_error = f"harbor run timed out after {self.timeout_sec}s"

        trajectories = self._parse_job(run_dir / "trials", batch, log_path, harbor_error)
        return EvaluationBatch(
            outputs=[t.stage_traces[-1]["output"] if t.stage_traces else "" for t in trajectories],
            scores=[t.score for t in trajectories],
            trajectories=trajectories if capture_traces else None,
        )

    def _parse_job(
        self,
        job_dir: Path,
        batch: list[str],
        log_path: Path,
        harbor_error: str,
    ) -> list[TaskTrajectory]:
        """One trajectory per batch task, aligned with the batch order.

        Score is the mean reward across attempts; reflection material comes
        from the lowest-reward attempt (failures carry the signal).
        """
        trials_by_task: dict[str, list[tuple[float, Path]]] = defaultdict(list)
        if job_dir.is_dir():
            for result_path in sorted(job_dir.glob("*/result.json")):
                trial_dir = result_path.parent
                try:
                    data = json.loads(result_path.read_text())
                except (OSError, ValueError):
                    continue
                task = data.get("task_name") or trial_dir.name.split("__")[0]
                trials_by_task[task].append((self._extract_reward(data, trial_dir), trial_dir))

        trajectories = []
        for task in batch:
            trials = trials_by_task.get(task) or next(
                (v for k, v in trials_by_task.items() if k.startswith(task) or task.startswith(k)),
                None,
            )
            if not trials:
                log_tail = _clip(log_path.read_text(), 2_000) if log_path.exists() else ""
                trajectories.append(TaskTrajectory(
                    task=task, score=0.0,
                    error=harbor_error or f"No trial output found for task. harbor.log tail:\n{log_tail}",
                ))
                continue

            score = sum(r for r, _ in trials) / len(trials)
            _, worst_dir = min(trials, key=lambda rt: rt[0])
            trajectories.append(self._load_trajectory(task, score, worst_dir))
        return trajectories

    @staticmethod
    def _extract_reward(result_data: dict, trial_dir: Path) -> float:
        rewards = (result_data.get("verifier_result") or {}).get("rewards") or {}
        try:
            if "reward" in rewards:
                return float(rewards["reward"])
            if rewards:
                return float(next(iter(rewards.values())))
            reward_txt = trial_dir / "verifier" / "reward.txt"
            if reward_txt.exists():
                return float(reward_txt.read_text().strip())
        except (TypeError, ValueError):
            pass
        return 0.0

    @staticmethod
    def _load_trajectory(task: str, score: float, trial_dir: Path) -> TaskTrajectory:
        def read(path: Path) -> str:
            try:
                return path.read_text()
            except OSError:
                return ""

        traces_raw = read(trial_dir / "agent" / "stage_traces.json")
        try:
            stage_traces = json.loads(traces_raw) if traces_raw else []
        except ValueError:
            stage_traces = []

        return TaskTrajectory(
            task=task,
            score=score,
            instruction=read(trial_dir / "agent" / "instruction.md"),
            stage_traces=stage_traces,
            test_output=_clip(read(trial_dir / "verifier" / "test-stdout.txt"), MAX_FEEDBACK_CHARS),
            error=_clip(read(trial_dir / "exception.txt"), 1_000),
        )

    # ------------------------------------------ reflective dataset

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories: list[TaskTrajectory] = list(eval_batch.trajectories or [])
        if self.use_judge:
            self._ensure_judge_feedback(trajectories)

        dataset: dict[str, list[dict[str, Any]]] = {}
        for component in components_to_update:
            records = []
            for traj in trajectories:
                trace = next(
                    (t for t in traj.stage_traces if t.get("stage") == component), None)
                records.append({
                    "Inputs": {
                        "task_instruction": _clip(traj.instruction, MAX_IO_CHARS),
                        "stage_input": _clip(trace.get("input", ""), MAX_IO_CHARS) if trace else "(stage did not run)",
                    },
                    "Generated Outputs": _clip(trace.get("output", ""), MAX_IO_CHARS) if trace else "(stage did not run)",
                    "Feedback": self._build_feedback(traj, component, trace),
                    "score": traj.score,
                    "task": traj.task,
                })
            dataset[component] = records
        return dataset

    def _build_feedback(self, traj: TaskTrajectory, component: str, trace: dict | None) -> str:
        lines = []
        if traj.score >= 1.0:
            lines.append(f"Task PASSED (reward={traj.score:.2f}).")
        else:
            lines.append(f"Task FAILED (reward={traj.score:.2f}).")

        if trace is None:
            lines.append(f"Stage '{component}' produced no trace — the pipeline may have crashed before reaching it.")
        elif traj.judge and component in traj.judge:
            j = traj.judge[component]
            lines.append(
                f"Stage judge score: {j.get('score', '?')}. "
                f"Missing: {j.get('missing', '')}. "
                f"Assessment: {j.get('assessment', '')}"
            )

        if traj.score < 1.0 and traj.test_output:
            lines.append(f"Verifier test output:\n{traj.test_output}")
        if traj.error:
            lines.append(f"Harness error:\n{traj.error}")
        return _clip("\n".join(lines), MAX_FEEDBACK_CHARS * 2)

    def _ensure_judge_feedback(self, trajectories: list[TaskTrajectory]) -> None:
        """Fill traj.judge for trajectories that have traces, via evaluator.py."""
        pending = [t for t in trajectories if t.judge is None and t.stage_traces]
        if not pending:
            return
        try:
            from benchmark.evaluator import evaluate_pipeline

            async def judge_all():
                return await asyncio.gather(
                    *(evaluate_pipeline(t.stage_traces, t.instruction) for t in pending),
                    return_exceptions=True,
                )

            for traj, result in zip(pending, asyncio.run(judge_all())):
                traj.judge = result if isinstance(result, dict) else {}
        except Exception as exc:  # judge is best-effort; reflection still works without it
            print(f"[gepa-adapter] stage judge unavailable ({exc}); continuing without it")
            for traj in pending:
                traj.judge = {}
