# svd_llm
Implementation of svd_llm for Qwen-like models (tested on Qwen 2.5 only).  
To compress, save and evaluate a model (for instance, Qwen 2.5 1.5B), you can run:
```
python main.py --model "Qwen/Qwen2.5-1.5B" --save_path "./output" --compress_mlp --compress_att_qkv --compress_att_out --compression_ratio 0.2 --batch_size 16 --max_whitening_samples 2048 --evaluate --eval_batch_size 16 --eval_tasks "wikitext|0"
```
You can also load an already compressed model for evaluation, so that you don't have to wait for another compression round:
```
python main.py --model "Qwen/Qwen2.5-1.5B" --save_path "./output" --compressed_model_path "./output/models/Qwen_Qwen2.5_1.5B/Qwen_Qwen2.5_1.5B_qkv_out_mlp_0.2_compressed.pt" --evaluate --eval_batch_size 16 --eval_tasks "wikitext|0"
```
Here's a list of all available possible arguments:  
- `--model`: LLM to load from huggingface (default = `Qwen/Qwen2.5-1.5B`)
- `--run_v2`: Run SVD-LLM V2 (default = `False`)
- `--dtype`: Weights dtype for original and compressed models (default = `float32`)
- `--compression_ratio`: Target compression ratio, 0.2 means removing about 20% of the params (default = `0.2`)
- `--calibration_dataset`: Calibration dataset, format is "datasetNameOrPath:split" (default = `tatsu-lab/alpaca:train`)
- `--max_length`: Maximum context length for the LLM during compression (default = `2048`)
- `--max_whitening_samples`: Number of calibration data samples used to calculate the whitening matrices. Please note that each sample is a concatenation of samples, until it reaches a length of `max_length` tokens (default = `256`)
- `--batch_size`: Batch size for data preprocessing and forward pass (default = `2`)
- `--seed`: Seed used while sampling the calibration data (default = `6363`)
- `--device`: Device used to load the model during forward pass and after compression (default = `cuda`)
- `--save_path`: Base path to save the whitening matrices and the compressed model checkpoints (default = `None`)
- `--whitening_mat_path`: Local path used to load the whitening matrices (default = `None`)
- `--use_compressed`: Use compressed model for evaluation (default = `False`)
- `--compressed_model_path`: Path to load an already compressed model (default = `None`)
- `--compress_mlp`: Compress MLP weights (default = `False`)
- `--compress_att_qkv`: Compress attention qkv projection matrices (default = `False`)
- `--compress_att_out`: Compress attention output projection matrices (default = `False`)
- `--het`: Assign heterogeneous compression ratio (default = `False`)
- `--group_criterion`: Criterion used to group weight matrices in heterogeneous setting. Possible values are `type`, `global` and `decoder` (default = `type`)
- `--group_patterns`: Groups used when grouping weight matrices by type, the pattern is "groupName1:weightType1,weightType2;groupName2:weightType1,weightType2;..." (default = `q_proj:self_attn.q_proj;k_proj:self_attn.k_proj;v_proj:self_attn.v_proj;o_proj:self_attn.o_proj;gate_proj:mlp.gate_proj;up_proj:mlp.up_proj;down_proj:mlp.down_proj`)
- `--score_metric`: Score metric to use for weight importance during heterogeneous ratio allocation. Possible values are `truncation` and `entropy` (default = `truncation`)
- `--hf_token`: Huggingface token used to download/upload models (default = `None`)
- `--evaluate`: Evaluate the model on a set of tasks (default = `False`)
- `--eval_sampling`: Use conditional sampling during evaluation (default = `False`)
- `--eval_batch_size`: Evaluation batch size (default = `8`)
- `--eval_tasks`: Evaluation tasks, the pattern is "taskName1,taskName2,...,taskNameK|numShots" or "taskName1,taskName2,...,taskNameK|numShots1,numShots2,...,numShotsK" (default = `wikitext|0`)
- `--eval_max_length`: Maximum context length for the LLM during evaluation (default = `4096`)
- `--eval_temperature`: Evaluation temperature (default = `0.7`)
- `--max_eval_tokens`: Maximum number of tokens to generate during evaluation (default = `256`)  

Two example scripts that can be used for evaluation of original and compressed model are available too (`eval-original.sh` and `eval-compressed.sh`).  

**WARNING**: Please note that the current implementation is not the most efficient in terms of VRAM usage. Thus, be careful increasing the batch size.
