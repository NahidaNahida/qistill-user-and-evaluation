# Qistill User and Evaluation

This repository contains the Skill User workflow for generating Python quantum programs from Quanbench task prompts. The workflow can optionally load a Skill bundle, expose its resources through mediated tools, and run Python Skill resources in a fixed Conda environment.

The `quanbench/` directory is an external benchmark checkout. Its files, including `quanbench/environment.yml`, are kept unchanged. The environment used by this repository is declared separately in the root `environment-qistill.yml`.

## Runtime environment

The supported runtime is:

- Conda environment: `qistill`
- Python: `3.10`
- Qiskit: `0.46.0`
- Qiskit Aer: `0.14.2`

Skill-resource scripts are executed through `conda run -n qistill`. The Agent does not receive unrestricted shell access and cannot select an arbitrary environment. It can use the fixed environment only through the Skill resource tools.

## Installation

Install Miniconda or Anaconda first, then open an Anaconda Prompt or a shell where the `conda` command is available.

Create the environment from the repository root:

```powershell
conda env create -f environment-qistill.yml
```

If an existing `qistill` environment should be updated instead, run:

```powershell
conda env update -n qistill -f environment-qistill.yml --prune
```

Activate it:

```powershell
conda activate qistill
```

Verify the interpreter and package versions:

```powershell
python --version
python -c "import qiskit, qiskit_aer; print(qiskit.__version__); print(qiskit_aer.__version__)"
```

The expected output is Python `3.10.x`, Qiskit `0.46.0`, and Qiskit Aer `0.14.2`.

The environment file installs Qiskit and Qiskit Aer through pip because these exact package versions are not reliably available from the Conda channels configured for this project. The resulting packages are still installed inside the `qistill` Conda environment.

## Running tests

With `qistill` activated, run the test suite from the repository root:

```powershell
python -m pytest -q
```

The live OpenRouter smoke test is disabled by default. To enable it, set `OPENROUTER_API_KEY` in the process environment and opt in explicitly:

```powershell
$env:QISTILL_LIVE_SMOKE = "1"
python -m pytest tests/test_live_task_loading_smoke.py -q
```

Do not commit API keys or local `.env` files. The `.gitignore` excludes common credential and local-runtime files.

## Running the workflow

The Python source root is `src`. From the repository root, set it on `PYTHONPATH` for direct use:

```powershell
$env:PYTHONPATH = "src"
```

The configured task/model/seed combinations can then be run with:

```python
from task_loading import run_configured_tasks

results = run_configured_tasks()
for result in results:
    print(result.artifact_path)
```

The default `execution_mode` is `"sequential"`, so tasks are processed one at a time. To use a worker pool, select `"parallel"`; each task still gets an independent workflow, prompt, tool trace, artifact, and runtime record. The returned results preserve the input task order.

```python
results = run_configured_tasks(
    execution_mode="parallel",
    max_workers=4,
)
```

Use sequential mode for debugging and deterministic inspection. Use parallel mode for throughput when provider rate limits and local resources allow it.

To use a Skill bundle, pass its directory through `skill_path`:

```python
from task_loading import run_configured_tasks

results = run_configured_tasks(skill_path="path/to/skill")
```

Quanbench task loading exposes only each task's `task_id` and prompt. The workflow then adds an explicit runtime notice to the Agent prompt stating that generated code must target Python 3.10 and Qiskit 0.46.0. This is necessary because the original Quanbench task prompt does not itself specify the Qiskit version.

Generated source artifacts are written under `data/generated_code/`, and runtime records are written under `data/runtime/`. These are local experiment outputs and are ignored by Git.
