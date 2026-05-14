# svd_llm

Implementation of **SVD-LLM** for Qwen-like models (tested primarily on Qwen 2.5 and LLama). This tool allows for post-training compression of Large Language Models using Singular Value Decomposition with whitening.

---

## Quick Start

### Compress and Evaluate

To compress a model (e.g., Qwen 2.5 7B), save the results, and run an immediate evaluation:

```bash
python main.py \
    --model "Qwen/Qwen2.5-7B" \
    --save_path "./output" \
    --compress_mlp \
    --compress_att_q \
    --compress_att_k \
    --compress_att_v \
    --compress_att_out \
    --compression_ratio 0.2 \
    --batch_size 16 \
    --max_whitening_samples 2048 \
    --evaluate \
    --eval_batch_size "auto" \
    --eval_tasks "wikitext|0"

```

### Evaluate a Compressed Model

If you have already compressed a model and saved the `.pt` checkpoint, you can bypass the compression stage:

```bash
python main.py \
    --model "Qwen/Qwen2.5-1.5B" \
    --compressed_model_path "./output/models/compressed_model.pt" \
    --evaluate \
    --eval_batch_size "auto" \
    --eval_tasks "wikitext|0"

```

---

## Arguments Reference

### Core Configuration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--model` | `str` | `Qwen/Qwen2.5-1.5B` | HF model identifier. |
| `--run_v2` | `flag` | `False` | Enable SVD-LLM V2. |
| `--dtype` | `str` | `float32` | Weights datatype for original/compressed models. |
| `--device` | `str` | `cuda` | Computing device. |
| `--hf_token` | `str` | `None` | Hugging Face token for restricted models. |

### Compression Settings

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--compression_ratio` | `float` | `0.2` | Target ratio (e.g., 0.2 removes ~20% of params). |
| `--ratio_scope` | `str` | `selected` | `selected`: ratio applies to chosen modules; `all`: ratio applies to the whole model (if only a subset of modules was chosen, they will have a higher average compression ratio). |
| `--compress_mlp` | `flag` | `False` | Compress MLP weights. |
| `--compress_att_q` | `flag` | `False` | Compress Attention Query projection. |
| `--compress_att_k` | `flag` | `False` | Compress Attention Key projection. |
| `--compress_att_v` | `flag` | `False` | Compress Attention Value projection. |
| `--compress_att_out` | `flag` | `False` | Compress Attention Output projection. |

### Whitening & Calibration

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--calibration_dataset` | `str` | `EleutherAI/wikitext_document_level:wikitext-2-raw-v1:train` | Format: `dataset:subset:split`. |
| `--max_length` | `int` | `2048` | Max context length during compression. |
| `--max_whitening_samples` | `int` | `256` | Number of samples for whitening matrix calculation. |
| `--batch_size` | `int` | `2` | Batch size for calibration forward pass. |
| `--whitening_only` | `flag` | `False` | Only calculate and save whitening matrices. |
| `--whitening_start_layer` | `int` | `0` | Start layer index for whitening. |
| `--whitening_end_layer` | `int` | `None` | End layer index for whitening. |

### Heterogeneous Compression

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--het` | `flag` | `False` | Enable heterogeneous compression ratio allocation. |
| `--score_metric` | `str` | `truncation` | Metric for weight importance (`truncation`, `entropy`). |
| `--group_criterion` | `str` | `type` | Grouping logic: `type`, `global`, or `decoder`. |
| `--bypass_early_layers` | `int` | `2` | Number of initial layers to exempt from compression. |
| `--bypass_ratio` | `float` | `0.0` | Compression ratio used for bypassed layers. |

### Evaluation

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--evaluate` | `flag` | `False` | Run evaluation tasks after loading/compressing. |
| `--eval_tasks` | `str` | `wikitext|0` | Task pattern: `task1,task2|shots`. |
| `--eval_batch_size` | `str` | `auto` | Batch size for evaluation. |
| `--eval_max_length` | `int` | `4096` | Max context length during evaluation. |
| `--max_eval_tokens` | `int` | `256` | Max tokens to generate during eval. |

---

> [!TIP]
> Use `generate_tables.py` to quickly generate latex/markdown tables with the evaluation results inside a folder.

> [!TIP]
> Use `generate_text.py` to quickly generate text with one or all models inside a directory (sequentially), using a predefined prompt. You can generate text with the original model too.