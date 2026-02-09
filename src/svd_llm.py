from typing import Dict, Optional
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
import torch
from torch.utils.data import DataLoader
from .utils import *

class LowRank(torch.nn.Module):
    def __init__(self, in_features, out_features, rank, bias):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W_v = nn.Linear(in_features, rank, bias)
        self.W_u = nn.Linear(rank, out_features, bias)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output =  self.W_u(self.W_v(input))
        return output

def get_whitening_matrices(
        model: Qwen2ForCausalLM,
        loader: DataLoader,
        layers_str: List[str], 
        layers_list: List,
        attributes: List[str],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    # Define pytorch forward hook to incrementally compute X^T @ X
    def hook(module, input, output):
        inp = input[0].detach().to(dtype=torch.float32) # Detach input activation from computational graph
        act = torch.matmul(inp.transpose(1,2), inp) # Calculate x^T @ x for the entire batch
        act_batch = torch.sum(act, dim=0) # Sum across batch dimension
        module.raw_xxt_matrix += act_batch
        del inp, act, act_batch
        torch.cuda.empty_cache()

    # Inizialize empty X^T @ X matrices and register forward hooks
    for layer, attr in tqdm(zip(layers_list, attributes), total=len(layers_list), desc="Registering hooks..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            layer_attr.raw_xxt_matrix = torch.zeros(layer_attr.in_features, layer_attr.in_features, device=device, dtype=torch.float32)
            layer_attr.register_forward_hook(hook)

    # Run inference on calibration data
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Running inference on calibration data..."):
            # REMOVE final_input before forward pass
            batch = {
                k: v.to(device)
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            model(**batch)

    for layer, attr in tqdm(zip(layers_list, attributes), total=len(layers_list), desc="Removing hooks..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            # Move X^T @ X matrix to CPU if it exists
            if hasattr(layer_attr, 'raw_xxt_matrix'):
                layer_attr.raw_xxt_matrix = layer_attr.raw_xxt_matrix.cpu()
            # Remove hook
            layer_attr._forward_hooks.clear()

    # Empty cache and move model to CPU
    torch.cuda.empty_cache()
    model = model.cpu()
    print("DEBUG: Cache emptied succesfully after model execution on calibration data")

    # Build whitening matrices applying Cholesky decomposition to X^T @ X
    whitening_matrices = {}
    for i, (layer, attr) in tqdm(enumerate(zip(layers_list, attributes)), total=len(layers_list), desc="Generating whitening matrices..."):
        layer_attr = getattr(layer, attr)
        if (isinstance(layer_attr, nn.Linear)):
            if hasattr(layer_attr, 'raw_xxt_matrix'):
                raw_xxt_matrix = layer_attr.raw_xxt_matrix.to(device, dtype=torch.float64)
                try:
                    whitening_matrix = torch.linalg.cholesky(raw_xxt_matrix)
                except Exception as e:
                    print("WARNING: eigen whitening_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_xxt_matrix)
                    raw_xxt_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_xxt_matrix.shape[0]).to(device)
                    whitening_matrix = torch.linalg.cholesky(raw_xxt_matrix)
                    eigenvalues = None
                    del eigenvalues
                whitening_matrices[layers_str[i]]=whitening_matrix.cpu()
                whitening_matrix = raw_xxt_matrix = layer_attr.raw_xxt_matrix = None
                del whitening_matrix, raw_xxt_matrix, layer_attr.raw_xxt_matrix
                torch.cuda.empty_cache()

    return whitening_matrices

# Compress model with SVD-LLM
def compress_svd_llm(
        model_name: str,
        ratio: float, 
        dataset: Dict,
        dtype: str = "bfloat16",
        batch_size: int = 32,
        seed: Optional[int] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        save_path: Optional[str] = None,
        whitening_mat_path: Optional[str] = None,
        compress_mlp: bool = False,
        compress_att_qkv: bool = False,
        compress_att_out: bool = False,
        hf_token: Optional[str] = None
):
    # Load model and tokenizer
    vram_usage("Before loading original model")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        use_safetensors=True,
        token=hf_token
    )
    vram_usage("After loading original model")
    # Avoid warning
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    # Preprocess calibration dataset
    print("=== DATASET PREPROCESSING ===")
    vram_usage("Before loading dataset")
    calibration_dataset = tokenize_dataset(
        dataset["dataset_name"],
        dataset["split"],
        tokenizer,
        dataset["max_samples"],
        dataset["seq_len"],
        batch_size,
        seed,
        save_path
    )
    calibration_dataloader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    vram_usage("After loading dataset")
    print("=== FINAL DATASET STRUCTURE ===")
    print(calibration_dataset)

    # Get list of layers of interest
    layers_str = generate_paths(
        compress_mlp, 
        compress_att_qkv, 
        compress_att_out, 
        layers_number=model.config.num_hidden_layers
    )
    layers_list, attributes = get_layers(model, layers_str, True)

    vram_usage("Before loading whitening matrices")
    if whitening_mat_path:
        print("DEBUG: Loading whitening matrices from disk...")
        whitening_matrices = torch.load(whitening_mat_path, map_location="cpu")
    else:
        print("=== WHITENING MATRICES GENERATION ===")
        whitening_matrices = get_whitening_matrices(
            model,
            calibration_dataloader,
            layers_str,
            layers_list,
            attributes,
            device
        )
        if save_path:
            print("DEBUG: Saving whitening matrices to disk...")
            save_path_whitening = save_path + "/whitening_matrices/"
            if not os.path.exists(save_path_whitening):
                os.makedirs(save_path_whitening)
            torch.save(whitening_matrices, save_path_whitening + 
                                           model_name.replace("/", "_").replace("-", "_") + 
                                           '_whitening_'+ 
                                           dataset["dataset_name"].replace("/", "_").replace("-", "_") + 
                                           '_' + 
                                           str(dataset["max_samples"]) + 
                                           '_' + 
                                           str(seed) + 
                                           '.pt')
    vram_usage("After loading whitening matrices")

    print("=== LLM COMPRESSION ===")
    vram_usage("Before performing layer replacement")
    rank_map = {}
    for i, (layer, attr) in tqdm(enumerate(zip(layers_list, attributes)), total=len(layers_list), desc="Running SVD and compressing layers..."):
        layer_attr = getattr(layer, attr)
        W = layer_attr.weight.data.to(device, dtype=torch.float32)
        whitening_matrix = whitening_matrices[layers_str[i]].to(device, dtype=torch.float64)

        # Compute the inverse of the whitening matrix
        try:
            whitening_matrix_inv = torch.linalg.inv(whitening_matrix)
        except Exception as e:
            print("WARNING: whitening_matrix is not full rank!")
            whitening_matrix += 1e-6 * torch.eye(whitening_matrix.shape[0], dtype=whitening_matrix.dtype).to(device)
            whitening_matrix_inv = torch.linalg.inv(whitening_matrix)

        # Cast to lower precision
        whitening_matrix = whitening_matrix.to(dtype=torch.float32)
        whitening_matrix_inv = whitening_matrix_inv.to(dtype=torch.float32)

        # Perform SVD on WS
        WS = torch.matmul(W, whitening_matrix)
        U, L, VT = torch.linalg.svd(WS, full_matrices = False)
        # Calculate the number of singular values to keep based on compression ratio
        num_sv_reduced = int((W.shape[0] * W.shape[1] * ratio) / (W.shape[1] + W.shape[0]))
        rank_map[layers_str[i]] = num_sv_reduced
        U_r = U[:, :num_sv_reduced]
        L_r = torch.diag(L[:num_sv_reduced])
        VT_r = torch.matmul(VT[:num_sv_reduced, :], whitening_matrix_inv)
        L_r_sqrt = torch.sqrt(L_r)
        # Compute the new weight matrices
        W_u = torch.matmul(U_r, L_r_sqrt).cpu().to(layer_attr.weight.dtype)
        W_v = torch.matmul(L_r_sqrt, VT_r).cpu().to(layer_attr.weight.dtype)
        # Replace layer with reduced one
        van = LowRank(
            layer_attr.in_features, 
            layer_attr.out_features, 
            num_sv_reduced, 
            layer_attr.bias is not None
        )
        van.W_u.weight.data = W_u
        van.W_v.weight.data = W_v
        if layer_attr.bias is not None:
            van.W_u.bias.data = layer_attr.bias.data

        setattr(layer,attr,van)
        W = whitening_matrix = whitening_matrix_inv = WS = U = L = VT = U_r = L_r = VT_r = L_r_sqrt = None
        del W, whitening_matrix, whitening_matrix_inv, WS, U, L, VT, U_r, L_r, VT_r, L_r_sqrt
        torch.cuda.empty_cache()

    vram_usage("After performing layer replacement")
    if save_path:
        print("DEBUG: Saving compressed model to disk...")
        # Create model directory
        save_path_model = save_path + \
                          "/models/" + \
                          model_name.replace("/", "_").replace("-", "_") + \
                          "/"
        if not os.path.exists(save_path_model):
            os.makedirs(save_path_model)
        # Save tokenizer
        tokenizer.save_pretrained(save_path_model)
        # Save model weights
        compress_mlp_str = "mlp_" if compress_mlp else ""
        compress_att_qkv_str = "qkv_" if compress_att_qkv else ""
        compress_att_out_str = "out_" if compress_att_out else ""
        torch.save({
            "state_dict": model.state_dict(),
            "rank_map": rank_map,
        }, save_path_model + 
           model_name.replace("/", "_").replace("-", "_") + 
           "_" + 
           compress_att_qkv_str + 
           compress_att_out_str + 
           compress_mlp_str + 
           str(round(1-ratio, 2)) + 
           "_compressed" + 
           ".pt")
        print("DEBUG: Compressed model saved succesfully")

    return model

def apply_lowrank(model, rank_map):
    """
    Replace MLP linear layers with LowRank modules.
    rank_map: dict with keys like 'model.layers.0.mlp.down_proj', 'model.layers.0.mlp.gate_proj', etc.
    """

    for layer_name, rank in rank_map.items():
        # Get the old layer
        layer_path = layer_name.split('.')[:-1]
        layer = model
        for sub_layer in layer_path:
            layer = getattr(layer, sub_layer)

        # Update the layer
        attr_name = layer_name.split('.')[-1]
        attr = getattr(layer, attr_name)
        setattr(
            layer,
            attr_name,
            LowRank(
                in_features=attr.in_features,
                out_features=attr.out_features,
                rank=rank,
                bias=attr.bias is not None
            )
        )