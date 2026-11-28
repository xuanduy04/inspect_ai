import builtins
import sys


def main() -> None:
    """Run a test program: `python -m custom_eval._pyexec <program.py> <token>`.

    Applies LiveCodeBench's `reliability_guard` (the human-eval guard), runs
    the program, and prints `<token>` only if it finished without raising
    (including `SystemExit`). The caller counts a pass only when the token is
    printed, so code that exits early never passes.

    The program runs in a fresh globals dict without `__name__`, as human-eval
    and EvalPlus run it (`exec(code, {})`), so an `if __name__ == "__main__":`
    block in the answer (its own tests, `unittest.main()`, a demo) does not
    run.
    """
    program_path, token = sys.argv[1], sys.argv[2]
    with open(program_path, encoding="utf-8") as f:
        program = f.read()
    write = sys.__stdout__.write if sys.__stdout__ is not None else None
    flush = sys.__stdout__.flush if sys.__stdout__ is not None else None

    from custom_eval._vendor.lcb.testing_util import reliability_guard

    reliability_guard()
    exit_ = builtins.exit
    try:
        code = compile(program, "program.py", "exec")
        exec(code, {"__builtins__": builtins})
    except BaseException as ex:  # noqa: B036
        print(f"{type(ex).__name__}: {ex}", file=sys.stderr)
        exit_(1)
    if write is not None and flush is not None:
        write(f"\n{token}\n")
        flush()


if __name__ == "__main__":
    main()
