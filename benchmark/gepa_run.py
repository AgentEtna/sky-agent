"""Optimize the pipeline's stage prompts with GEPA.

Runs gepa.optimize() over the stage system_prompts in pipeline_spec.yaml,
scoring candidates by Harbor task rewards (see gepa_adapter.py). Topology,
tools, and turn budgets are untouched — those belong to the meta-agent.

Usage:
    uv run python -m benchmark.gepa_run                        # triage set, default budget
    uv run python -m benchmark.gepa_run --max-metric-calls 60  # smaller budget
    uv run python -m benchmark.gepa_run --write-back           # commit best prompts to the spec

Requires OPENAI_API_KEY (pipeline stages, judge, and reflection LM).

Budget intuition: one metric call = one Harbor task rollout. With the default
10-task triage set, every full validation sweep costs 10 calls, and each
mutation costs `--minibatch-size` calls before (maybe) a sweep. 60-150 calls
is a sensible range; expect minutes per rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from benchmark.gepa_adapter import (
    BASE_SPEC_PATH,
    REPO_ROOT,
    TRIAGE_TASKS,
    HarborPipelineAdapter,
)


def _reflection_lm(model: str):
    """gepa LanguageModel protocol: (prompt | messages) -> str, via OpenAI."""
    from openai import OpenAI

    client = OpenAI()

    def call(prompt):
        messages = prompt if isinstance(prompt, list) else [
            {"role": "user", "content": str(prompt)}]
        response = client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content or ""

    return call


class _LiteralDumper(yaml.SafeDumper):
    pass


def _repr_multiline_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralDumper.add_representer(str, _repr_multiline_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize pipeline stage prompts with GEPA")
    parser.add_argument("--tasks", nargs="*", default=TRIAGE_TASKS,
                        help="Task names to optimize against (default: the 10-task triage set)")
    parser.add_argument("--dataset", default="terminal-bench@2.0")
    parser.add_argument("--max-metric-calls", type=int, default=100,
                        help="Total rollout budget (1 call = 1 task rollout)")
    parser.add_argument("--reflection-model", default="gpt-5",
                        help="Model that proposes prompt mutations")
    parser.add_argument("--minibatch-size", type=int, default=3,
                        help="Tasks per reflection minibatch")
    parser.add_argument("--n-concurrent", type=int, default=10)
    parser.add_argument("--n-attempts", type=int, default=1,
                        help="Attempts per task per evaluation (score = mean reward)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the evaluator.py stage judge in reflection feedback")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where rollouts and GEPA state go (default: jobs/gepa/<timestamp>)")
    parser.add_argument("--write-back", action="store_true",
                        help="Write the best prompts into benchmark/pipeline_spec.yaml")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — required for pipeline stages, "
                 "the stage judge, and the reflection LM.")

    output_dir = args.output_dir or (
        REPO_ROOT / "jobs" / "gepa" / datetime.now().strftime("%Y%m%d-%H%M%S"))
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = yaml.safe_load(BASE_SPEC_PATH.read_text())
    seed_candidate = {
        stage["id"]: stage["system_prompt"].strip()
        for stage in spec["pipeline"]["stages"]
    }
    print(f"[gepa] components: {list(seed_candidate)}")
    print(f"[gepa] tasks ({len(args.tasks)}): {', '.join(args.tasks)}")
    print(f"[gepa] budget: {args.max_metric_calls} rollouts -> {output_dir}")

    adapter = HarborPipelineAdapter(
        dataset=args.dataset,
        output_dir=output_dir,
        n_concurrent=args.n_concurrent,
        n_attempts=args.n_attempts,
        use_judge=not args.no_judge,
    )

    import gepa

    result = gepa.optimize(
        seed_candidate=seed_candidate,
        trainset=list(args.tasks),
        adapter=adapter,
        reflection_lm=_reflection_lm(args.reflection_model),
        reflection_minibatch_size=args.minibatch_size,
        max_metric_calls=args.max_metric_calls,
        run_dir=str(output_dir / "gepa_state"),
        display_progress_bar=True,
        raise_on_exception=False,
        seed=args.seed,
    )

    baseline = result.val_aggregate_scores[0]
    best = result.val_aggregate_scores[result.best_idx]
    print(f"\n[gepa] baseline avg reward: {baseline:.3f}")
    print(f"[gepa] best avg reward:     {best:.3f} "
          f"(candidate {result.best_idx} of {result.num_candidates})")

    best_candidate = dict(result.best_candidate)
    (output_dir / "best_candidate.json").write_text(json.dumps(best_candidate, indent=2))

    optimized_spec = adapter.render_spec(best_candidate)
    optimized_spec_path = output_dir / "optimized_pipeline_spec.yaml"
    optimized_spec_path.write_text(
        yaml.dump(optimized_spec, Dumper=_LiteralDumper, sort_keys=False))
    print(f"[gepa] optimized spec: {optimized_spec_path}")

    if args.write_back:
        BASE_SPEC_PATH.write_text(
            yaml.dump(optimized_spec, Dumper=_LiteralDumper, sort_keys=False))
        print(f"[gepa] wrote best prompts into {BASE_SPEC_PATH}")
    elif best > baseline:
        print("[gepa] rerun with --write-back (or copy the optimized spec) to adopt the prompts")


if __name__ == "__main__":
    main()
