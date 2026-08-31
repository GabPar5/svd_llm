import argparse
import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Set

DEFAULT_BASE = Path("args/base_args.json")
DEFAULT_EXPERIMENTS = Path("args/experiments.json")
MAIN_SCRIPT = Path("main.py")

# A stage file carries these until the preceding stage's gate resolves them, so
# they are caught here rather than an hour into a run
PLACEHOLDER_MARKER = "__"

# The four `build_run_name` parameters spelled differently from the main.py flag
# that feeds them
RUN_NAME_ALIASES = {
    "model_name": "model",
    "ratio": "compression_ratio",
    "heterogeneous": "het",
    "is_v2": "run_v2",
}

def load_json(path: Path) -> Any:
    if not path.exists():
        available = sorted(sibling.name for sibling in path.parent.glob("experiments_*.json"))
        known = "\n".join(f"  {name}" for name in available)
        raise SystemExit(
            f"No such file: {path}\n"
            + (f"\nStage files in {path.parent}:\n{known}" if available else ""),
        )

    with path.open() as json_file:
        return json.load(json_file)

def find_placeholders(exp: Dict[str, Any]) -> List[str]:
    """Report the arguments of one run that still hold an unresolved gate value"""
    return [
        f"{key} {value}" for key, value in exp.items()
        if isinstance(value, str) and PLACEHOLDER_MARKER in value
    ]

def build_command(base: Dict[str, Any], exp: Dict[str, Any]) -> List[str]:
    """Compile the argument dictionaries into a subprocess-compatible command list"""
    cmd = [ sys.executable, "main.py" ]

    # Merge base and experiment args (exp overwrites base keys if they clash)
    merged = {**base, **exp}

    for key, value in merged.items():
        if isinstance(value, bool):
            # For argparse action='store_true', only append the flag if True
            if value:
                cmd.append(key)
        elif value is not None:
            cmd.extend([ key, str(value) ])
    return cmd

def argparse_defaults(script: Path = MAIN_SCRIPT) -> Dict[str, Any]:
    """
    The default of every main.py flag whose default is a literal.

    main.py builds its parser under `if __name__ == "__main__"`, so it cannot be
    imported and asked. Reading its source keeps that argparse block the single
    source of truth instead of copying its defaults here to drift out of step.
    A default that is an enum member or a module constant is left out, because
    `build_run_name` carries the same value in its own signature and answers for
    it when the argument is omitted.
    """
    defaults: Dict[str, Any] = {}

    if not script.exists():
        return defaults

    for node in ast.walk(ast.parse(script.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue

        is_add_argument = (
            getattr(node.func, "attr", None) == "add_argument"
            and bool(node.args)
        )
        if not is_add_argument:
            continue

        flag_node = node.args[0]
        if not isinstance(flag_node, ast.Constant):
            continue

        flag = str(flag_node.value).lstrip("-")
        keywords = { word.arg: word.value for word in node.keywords if word.arg }
        action = keywords.get("action")

        if isinstance(action, ast.Constant) and action.value == "store_true":
            defaults[flag] = False
        elif "default" in keywords:
            try:
                defaults[flag] = ast.literal_eval(keywords["default"])
            except (ValueError, TypeError, SyntaxError):
                continue

    return defaults

class Settings(NamedTuple):
    """One run's arguments, read the way main.py will read them"""
    merged: Dict[str, Any]
    defaults: Dict[str, Any]

    def get(self, flag: str, fallback: Any = None) -> Any:
        if f"--{flag}" in self.merged:
            return self.merged[f"--{flag}"]
        return self.defaults.get(flag, fallback)

    def has(self, flag: str) -> bool:
        return f"--{flag}" in self.merged or flag in self.defaults

def resolved_run_name(settings: Settings) -> Optional[str]:
    """
    The basename main.py will give this run's artifacts, or None when the
    arguments alone do not decide it.

    Mirrors the three entry paths main.py picks between, plus the rename
    `--update_taw_only` applies to the checkpoint it started from.
    """
    from src.utils import build_run_name, sanitize_model_name

    model = settings.get("model")
    if not model:
        return None

    if not settings.get("use_compressed", False):
        return sanitize_model_name(str(model))

    checkpoint = settings.get("compressed_model_path")
    if checkpoint:
        stem = os.path.splitext(os.path.basename(str(checkpoint)))[0]
        if not settings.get("update_taw_only", False):
            return stem
        # The update writes a checkpoint of its own and the evaluation follows it
        return f"{stem}_sequpd_{settings.get('sequential_update_method')}"

    arguments = {}
    for parameter in inspect.signature(build_run_name).parameters:
        flag = RUN_NAME_ALIASES.get(parameter, parameter)
        if settings.has(flag):
            arguments[parameter] = settings.get(flag)

    try:
        return build_run_name(**arguments)
    except TypeError:
        # A parameter this build_run_name requires has no value here, so the name
        # cannot be known and the run has to happen
        return None

def requested_tasks(eval_tasks: Any) -> Set[str]:
    """The task names one entry asks its evaluation for, from `"t1,t2|shots"`"""
    if not isinstance(eval_tasks, str):
        return set()
    return { task.strip() for task in eval_tasks.split("|")[0].split(",") if task.strip() }

def evaluated_tasks(results_path: Path) -> Set[str]:
    """What an evaluation JSON already holds, empty when there is none to read"""
    if not results_path.exists():
        return set()

    try:
        with results_path.open(encoding="utf-8") as results_file:
            return set(json.load(results_file).get("results", {}))
    except (json.JSONDecodeError, OSError):
        return set()

class Completion(NamedTuple):
    path: Path
    done: Set[str]
    missing: Set[str]

    @property
    def complete(self) -> bool:
        return not self.missing

def completion_of(base: Dict[str, Any], exp: Dict[str, Any], defaults: Dict[str, Any]) -> Optional[Completion]:
    """
    What this run has already evaluated, or None when that cannot be established.

    None is the answer for a run that writes no evaluation to compare against and
    for one whose name does not resolve. Callers read it as "not complete", so an
    undecidable run is executed rather than silently skipped: a wrong skip loses
    data, a wrong run only costs time.
    """
    settings = Settings({**base, **exp}, defaults)

    writes_no_evaluation = (
        not settings.get("evaluate", False)
        or settings.get("whitening_only", False)
    )
    if writes_no_evaluation:
        return None

    wanted = requested_tasks(settings.get("eval_tasks"))
    if not wanted:
        return None

    name = resolved_run_name(settings)
    if name is None:
        return None

    from src.utils import sanitize_model_name

    results_path = (
        Path(str(settings.get("save_path", "./output")))
        / "eval"
        / sanitize_model_name(str(settings.get("model")))
        / f"{name}.json"
    )
    done = evaluated_tasks(results_path)

    return Completion(path=results_path, done=done & wanted, missing=wanted - done)

def describe(completion: Optional[Completion]) -> str:
    """The one-line reason a run is being executed or skipped"""
    if completion is None:
        return "no evaluation to compare against"
    if completion.complete:
        return f"already evaluated: {', '.join(sorted(completion.done))}"
    if completion.done:
        return f"missing: {', '.join(sorted(completion.missing))}"
    return "not yet evaluated"

def main() -> None:
    parser = argparse.ArgumentParser(description="Run one stage of the experiment grid")
    parser.add_argument(
        'experiments',
        nargs='?',
        type=Path,
        default=DEFAULT_EXPERIMENTS,
        help='Stage file holding one argument dictionary per run',
    )
    parser.add_argument(
        '--base',
        type=Path,
        default=DEFAULT_BASE,
        help='Arguments shared by every run of the stage',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print the commands the stage would run, without running them',
    )
    parser.add_argument(
        '--skip_completed',
        action='store_true',
        help=(
            'Skip a run whose evaluation JSON already holds every task it asks for. '
            'Reported either way, so leaving this off still says what would be skipped'
        ),
    )
    args = parser.parse_args()

    base_args = load_json(args.base)
    experiments = load_json(args.experiments)

    unresolved = [ (i, find_placeholders(exp)) for i, exp in enumerate(experiments, 1) ]
    unresolved = [ (i, found) for i, found in unresolved if found ]

    if unresolved:
        print(f"[ERROR] {args.experiments} still holds unresolved gate values:")
        for i, found in unresolved:
            print(f"  run {i}: {', '.join(found)}")
        print("\nFill them from the preceding stage's results, see EXPERIMENTS.md")
        sys.exit(1)

    defaults = argparse_defaults()
    completions = [ completion_of(base_args, exp, defaults) for exp in experiments ]
    done = [ i for i, found in enumerate(completions, 1) if found is not None and found.complete ]

    total_runs = len(experiments)
    mode = "dry run" if args.dry_run else "consecutive runs"
    print(f"Initializing orchestrator: {total_runs} {mode} scheduled from {args.experiments}")

    if done and not args.skip_completed:
        print(f"{len(done)} of {total_runs} already complete -- pass --skip_completed to skip them")
    elif done:
        print(f"Skipping {len(done)} of {total_runs} already complete")
    print()

    for i, (exp, completion) in enumerate(zip(experiments, completions), 1):
        cmd = build_command(base_args, exp)
        skipping = args.skip_completed and completion is not None and completion.complete
        verb = "SKIPPING" if skipping else ("PREVIEWING" if args.dry_run else "EXECUTING")

        print("=" * 80)
        print(f"{verb} RUN {i}/{total_runs}  ({describe(completion)})")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80 + "\n")

        if skipping or args.dry_run:
            continue

        try:
            # check=True raises a CalledProcessError if the script crashes
            subprocess.run(cmd, check=True)
            print(f"\n[SUCCESS] Run {i} completed successfully\n")

        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Run {i} failed with exit code {e.returncode}")
            print("Continuing to the next experiment in the queue...\n")
            # Replace the print above with sys.exit(1) to halt on the first failure

        except KeyboardInterrupt:
            print("\n[INTERRUPT] Orchestrator manually stopped by user. Exiting")
            sys.exit(0)

if __name__ == "__main__":
    main()
