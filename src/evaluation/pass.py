def quanbench_correctness_pass(
    generated_code_path: str,
    reference_data_path: str,
    timeout: float = 50.0,
) -> bool:
    """
    Evaluates whether the generated code is functionally equivalent to the reference code based on 
    the QuanBench assertions.
    """

    import os

    from quanbench.Quanbench_eval.data import read_problems
    from quanbench.Quanbench_eval.execution import check_correctness

    if not isinstance(generated_code_path, str) or not generated_code_path.strip():
        raise TypeError("generated_code_path must be a non-empty string.")
    if not isinstance(reference_data_path, str) or not reference_data_path.strip():
        raise TypeError("reference_data_path must be a non-empty string.")
    if not os.path.isfile(generated_code_path):
        raise FileNotFoundError(f"Generated code file does not exist: {generated_code_path}")
    if not os.path.isfile(reference_data_path):
        raise FileNotFoundError(f"Reference dataset does not exist: {reference_data_path}")
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive number.")

    with open(generated_code_path, "r", encoding="utf-8") as generated_file:
        generated_code = generated_file.read()
    if not generated_code.strip():
        return False

    task_id = os.path.splitext(os.path.basename(generated_code_path))[0]
    problems = read_problems(reference_data_path)
    if task_id not in problems:
        raise KeyError(
            f"Task {task_id!r} derived from the generated filename is not present "
            f"in {reference_data_path}."
        )

    # True and False describe only the generated artifact's result against the
    # matching QuanBench assertions. Input, loading, and task-resolution errors
    # are raised above instead of being mislabeled as generated-code failures.
    result = check_correctness(problems[task_id], generated_code, timeout=float(timeout))
    return result.get("passed") is True
 

def compilation_pass(generated_code_path: str) -> bool:
    """
    Evaluates whether the generated code compiles successfully.
    """
    import py_compile

    if not isinstance(generated_code_path, str) or not generated_code_path.strip():
        raise TypeError("generated_code_path must be a non-empty string.")

    try:
        py_compile.compile(generated_code_path, doraise=True)
    except py_compile.PyCompileError:
        return False
    return True

def execution_pass(generated_code_path: str) -> bool:
    """
    Evaluates whether the generated code can be executed successfully without crashing.
    """
    import os
    import subprocess

    if not isinstance(generated_code_path, str) or not generated_code_path.strip():
        raise TypeError("generated_code_path must be a non-empty string.")
    if not os.path.isfile(generated_code_path):
        raise FileNotFoundError(f"Generated code file does not exist: {generated_code_path}")

    try:
        completed = subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                "qistill",
                "python",
                generated_code_path,
            ],
            cwd=os.path.dirname(os.path.abspath(generated_code_path)),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return completed.returncode == 0
