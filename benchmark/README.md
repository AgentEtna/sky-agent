# AutoSwarm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/arteemg/autoswarm?style=social)](https://github.com/arteemg/autoswarm)
[![Forks](https://img.shields.io/github/forks/arteemg/autoswarm?style=social)](https://github.com/arteemg/autoswarm)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?logo=docker&logoColor=white)](https://docker.com)

<p align="center">
  <br>
  <a href="https://discord.gg/9ggSRAFGKQ">
    <img src="https://img.shields.io/badge/Discord-Join%20Community-5865F2?logo=discord&logoColor=white" alt="Discord" />
  </a>
   <br>
  <img src="../assets/superagent_solo.gif" alt="AutoSwarm" width="400">
</p>


AutoSwarm is a benchmark harness over [Harbor](https://github.com/laude-institute/harbor) tasks. A meta-agent edits stage prompts, tools, turn budgets, and pipeline structure to hill-climb on `passed` tasks.

## Install

```bash
pip install -e .            # editable install from the repo
```

Requires Python 3.12+.




## Running the meta-agent

Point your coding agent at the repo and prompt:

```
Read benchmark/program_pipeline.md and let's kick off a new experiment!
```

The meta-agent will read the directive, inspect `benchmark/pipeline_spec.yaml`, run the benchmark, score each stage with `benchmark/evaluator.py`, edit the topology, and iterate.

<img src="../assets/sample_results.png" alt="Sample results" width="600">

## Task format

Tasks follow [Harbor's format](https://harborframework.com/docs/tasks):

```text
tasks/my-task/
  task.toml           -- config (timeouts, metadata)
  instruction.md      -- prompt sent to the agent
  tests/
    test.sh           -- entry point, writes /logs/reward.txt
    test_outputs.py   -- verification (deterministic or LLM-as-judge)
  environment/
    Dockerfile        -- task container image for Harbor
```

## License

MIT
