import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_BASE = Path("args/base_args.json")
DEFAULT_EXPERIMENTS = Path("args/experiments.json")

# A stage file carries these until the preceding stage's gate resolves them, so
# they are caught here rather than an hour into a run
PLACEHOLDER_MARKER = "__"

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

    total_runs = len(experiments)
    mode = "dry run" if args.dry_run else "consecutive runs"
    print(f"Initializing orchestrator: {total_runs} {mode} scheduled from {args.experiments}\n")

    for i, exp in enumerate(experiments, 1):
        cmd = build_command(base_args, exp)

        print("=" * 80)
        print(f"{'PREVIEWING' if args.dry_run else 'EXECUTING'} RUN {i}/{total_runs}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80 + "\n")

        if args.dry_run:
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
