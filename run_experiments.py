import subprocess
import sys
import json
from typing import List, Dict, Any

# 1. BASE CONFIGURATION
# Define the arguments that remain constant across all your runs.
with open("args/base_args.json") as json_file:
    BASE_ARGS = json.load(json_file)

# 2. EXPERIMENT GRID
# Define a list of dictionaries for the parameters you want to sweep. 
# These will override the BASE_ARGS for their specific run.
with open("args/experiments.json") as json_file:
    EXPERIMENTS = json.load(json_file)

def build_command(base: Dict[str, Any], exp: Dict[str, Any]) -> List[str]:
    """Compiles the argument dictionary into a subprocess-compatible command list."""
    cmd = [sys.executable, "main.py"]
    
    # Merge base and experiment args (exp overwrites base keys if they clash)
    merged = {**base, **exp}
    
    for key, value in merged.items():
        if isinstance(value, bool):
            # For argparse action='store_true', only append the flag if True
            if value:
                cmd.append(key)
        elif value is not None:
            cmd.extend([key, str(value)])
    return cmd

def main():
    total_runs = len(EXPERIMENTS)
    print(f"Initializing orchestrator: {total_runs} consecutive runs scheduled.\n")
    
    for i, exp in enumerate(EXPERIMENTS, 1):
        cmd = build_command(BASE_ARGS, exp)
        
        print("=" * 80)
        print(f"EXECUTING RUN {i}/{total_runs}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 80 + "\n")
        
        try:
            # check=True raises a CalledProcessError if the script crashes
            subprocess.run(cmd, check=True)
            print(f"\n[SUCCESS] Run {i} completed successfully.\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n[ERROR] Run {i} failed with exit code {e.returncode}.")
            print("Continuing to the next experiment in the queue...\n")
            # If you prefer the orchestrator to halt entirely on a failure, 
            # replace the print above with: sys.exit(1)
            
        except KeyboardInterrupt:
            print("\n[INTERRUPT] Orchestrator manually stopped by user. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    main()