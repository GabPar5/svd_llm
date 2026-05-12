import subprocess
import sys
from typing import List, Dict, Any

# 1. BASE CONFIGURATION
# Define the arguments that remain constant across all your runs.
BASE_ARGS = {
    "--model": "Qwen/Qwen2.5-32B", 
    "--use_compressed": True,
    "--dtype": "bfloat16",
    "--device": "cuda",
    "--calibration_dataset": "EleutherAI/wikitext_document_level:wikitext-2-raw-v1:train",
    "--max_length": 2048,
    "--seed": 3,
    "--save_path": "./output",
    "--whitening_mat_path": "./output/whitening_matrices",
    "--run_v2": True,
    "--compress_mlp": True,
    "--compress_att_qkv": True,
    "--compress_att_out": True,
    "--evaluate": True,
    "--eval_tasks": "wikitext,openbookqa,winogrande,hellaswag,arc_easy,piqa,truthfulqa_mc2|0,0,0,0,0,0,0",
    "--eval_max_length": 2048,
    "--max_eval_tokens": 256,
    "--eval_batch_size": 32
}

# 2. EXPERIMENT GRID
# Define a list of dictionaries for the parameters you want to sweep. 
# These will override the BASE_ARGS for their specific run.
EXPERIMENTS = [
    # Run 1: Baseline Homogeneous Compression
    {
        "--compression_ratio": 0.2, 
        "--het": False
    },
    # Run 2: Heterogeneous Type Truncation (Bypassing early layers)
    {
        "--compression_ratio": 0.2, 
        "--het": True, 
        "--group_criterion": "type",
        "--score_metric": "truncation",
        "--bypass_early_layers": 8,
        "--bypass_ratio": 0.0
    },
    # Run 3: Heterogeneous Decoder Truncation (Bypassing early layers)
    {
        "--compression_ratio": 0.2, 
        "--het": True, 
        "--group_criterion": "decoder",
        "--score_metric": "truncation",
        "--bypass_early_layers": 8,
        "--bypass_ratio": 0.0
    },
    # Run 4: Heterogeneous Type Entropy (Bypassing early layers)
    {
        "--compression_ratio": 0.2, 
        "--het": True, 
        "--group_criterion": "type",
        "--score_metric": "entropy",
        "--bypass_early_layers": 8,
        "--bypass_ratio": 0.0
    },
    # Run 5: Heterogeneous Decoder Entropy (Bypassing early layers)
    {
        "--compression_ratio": 0.2, 
        "--het": True, 
        "--group_criterion": "decoder",
        "--score_metric": "entropy",
        "--bypass_early_layers": 8,
        "--bypass_ratio": 0.0
    },
    # Run 6: Heterogeneous Decoder Entropy (no bypassed layers)
    {
        "--compression_ratio": 0.2, 
        "--het": True, 
        "--group_criterion": "decoder",
        "--score_metric": "entropy",
        "--bypass_early_layers": -1,
        "--bypass_ratio": 0.0
    },
]

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