# svd_llm
Implementation of svd_llm for Qwen-like models (tested on Qwen 2.5 only).  
To compress, save and evaluate a model (for instance, Qwen 2.5 1.5B), you can run:
```
python main.py --model "Qwen/Qwen2.5-1.5B" --save_path "./output" --compress_mlp --compress_att_qkv --compress_att_out --compression_ratio 0.2 --batch_size 16 --max_whitening_samples 2048 --evaluate --eval_batch_size 16 --eval_benchmarks "wikitext:103:document_level|0"
```
You can also load an already compressed model for evaluation, so that you don't have to wait for another compression round:
```
python main.py --model "Qwen/Qwen2.5-1.5B" --save_path "./output" --compressed_model_path "./output/models/Qwen_Qwen2.5_1.5B/Qwen_Qwen2.5_1.5B_qkv_out_mlp_0.2_compressed.pt" --evaluate --eval_batch_size 16 --eval_benchmarks "wikitext:103:document_level|0"
```
Here's a list of all available possible arguments:  
- `--model`: LLM to load from huggingface (default = `Qwen/Qwen2.5-1.5B`)
- `--dtype`: Weights dtype for original and compressed models (default = `bfloat16`)
- `--compression_ratio`: Target compression ratio, 0.2 means only keeping about 20% of the params (default = `0.2`)
- `--calibration_dataset`: Calibration dataset, format is "datasetNameOrPath:split" (default = `tatsu-lab/alpaca:train`)
- `--max_whitening_samples`: Number of calibration data samples used to calculate the whitening matrices (default = `1024`)
- `--batch_size`: Batch size for data preprocessing and forward pass (default = `16`)
- `--seed`: Seed used while sampling the calibration data (default = `6363`)
- `--device`: Device used to load the model during forward pass and after compression (default = `cuda`)
- `--save_path`: Base path to save the whitening matrices and the compressed model checkpoints. (default = `None`)
- `--whitening_mat_path`: Local path used to load the whitening matrices (default = `None`)
- `--use_compressed`: Use compressed model for evaluation (default = `False`)
- `--compressed_model_path`: Path to load an already compressed model (default = `None`)
- `--compress_mlp`: Compress MLP weights (default = `False`)
- `--compress_att_qkv`: Compress attention qkv matrices (default = `False`)
- `--compress_att_out`: Compress attention output projection matrices (default = `False`)
- `--hf_token`: Huggingface token used to download/upload models (default = `None`)
- `--evaluate`: Evaluate the model on a set of benchmarks (default = `False`)
- `--eval_batch_size`: Evaluation batch size (default = `1`)
- `--eval_benchmarks`: Evaluation benchmarks, the pattern is "benchmarkName:taskName|numShots,..." (default = `ethics:commonsense|0`)
- `--eval_temperature`: Evaluation temperature (default = `0.7`)
- `--max_eval_tokens`: Max number of tokens to generate during evaluation (default = `1024`)  


**WARNING**: Please note that the current implementation is not the most efficient in terms of VRAM usage. Thus, be careful increasing the batch size.
