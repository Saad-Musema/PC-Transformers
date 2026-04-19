import torch
from tokenizers import Tokenizer
from predictive_coding.config import GPTConfig
from utils.model_utils import load_model, decode_ids
from utils.config_utils import load_best_config
from utils.model_utils import set_seed
import torch.nn.functional as F
import torch.distributed as dist
from utils.device_utils import setup_device
import argparse
from data_preparation.config import vocab_size, tokenizer_path

"""
This script generates text using the trained predictive coding transformer model.
It takes a prompt, generates new tokens, and prints the prompt, target, and generated text.

Usage: torchrun --nproc-per-node=<NUM_GPU> generate_text.py

"""
local_rank, device, use_distributed = setup_device()

def generate_text(model, config, input_ids, max_new_tokens, temperature, device = None, use_cache=True):
    model.eval()
    base_model = model.module if hasattr(model, 'module') else model
    
    if input_ids is None:
        start_token = getattr(config, "start_token_id", 0)  
        input_tensor = torch.tensor([start_token], device=device).unsqueeze(0)
    else:
        input_tensor = input_ids.unsqueeze(0).to(device)

    # Clear KV cache at the start
    if use_cache:
        for module in base_model.modules():
            if hasattr(module, 'clear_kv_cache'):
                module.clear_kv_cache()
    
    generated_tokens = []
    
    # Process prompt once, then only feed last token
    with torch.no_grad():
        model(input_tensor, input_tensor, use_kv_cache=True)

    current_input = input_tensor[:, -1:]  # S=1 always
    for step in range(max_new_tokens):
        # For first token or with cache, pass full or last token
        current_input = input_tensor[:, -config.block_size:] if input_tensor.size(1) > config.block_size else input_tensor

        if hasattr(base_model, "set_debug_context"):
            base_model.set_debug_context(mode="generate", batch_idx=step, generation_step=step)

        logits = model(current_input, current_input, use_kv_cache=use_cache)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        generated_tokens.append(next_token.item())
        input_tensor = torch.cat((input_tensor, next_token), dim=1)
        
        if next_token.item() == getattr(config, 'eos_token_id', None):
            break
                      
    return input_tensor[0] 

def text_generation(model, config, device = None,  max_samples=2, max_new_tokens=200, use_cache = True):
    decoded_preds = []

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    
    for sample_idx in range(max_samples):
        generated_ids = generate_text(model, config, input_ids=None, max_new_tokens=max_new_tokens, temperature=0.7, device=device, use_cache=use_cache)
        generated_str = decode_ids(tokenizer, generated_ids.tolist(), stop_at_eos=True)

        print(f"\n[Sample {sample_idx + 1}]")
        print(f"[GENERATED]: {generated_str}")

        decoded_preds.append(generated_str)

    return decoded_preds

def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--flash', action='store_true', help='Enable FlashAttention for attention layers')
    parser.add_argument('--debug-layers', action='store_true', help='Log per-layer predictive-coding stats')
    parser.add_argument('--debug-max-batches', type=int, default=5, help='Number of generation steps to trace when layer debug is enabled')
    parser.add_argument('--debug-max-steps', type=int, default=None, help='Maximum predictive-coding iterations to trace')
    parser.add_argument('--debug-rank', type=int, default=0, help='Distributed rank that emits layer debug logs')
    args = parser.parse_args()

    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    print(f"[Rank {local_rank}] Using device: {device}")
    
    best_config = load_best_config()

    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = best_config["block_size"],
        lr = best_config["peak_learning_rate"],
        inference_lr = best_config["inference_lr"],
        peak_learning_rate = best_config["peak_learning_rate"],
        warmup_steps = best_config["warmup_steps"],
        n_embed = best_config["n_embed"],
        dropout = best_config["dropout"],
        T = best_config["T"],
        num_heads = best_config["num_heads"],
        n_blocks = best_config["n_blocks"],
        batch_size = best_config["batch_size"],
        num_epochs = best_config["num_epochs"], 
        update_bias = best_config["update_bias"],
        internal_energy_fn_name=best_config["internal_energy_fn_name"],
        output_energy_fn_name=best_config["output_energy_fn_name"],
        combined_internal_weight=best_config["combined_internal_weight"],
        combined_output_weight=best_config["combined_output_weight"],
        use_flash_attention=args.flash or best_config["use_flash_attention"],
        alpha = best_config["alpha"]    
    )
    
    model_path = "checkpoints/final_model.pt"
    model = load_model(model_path, config)
    model = model.to(device)
    model.configure_layer_debug(
        enabled=args.debug_layers and (not dist.is_initialized() or dist.get_rank() == args.debug_rank),
        max_batches=args.debug_max_batches,
        max_steps=args.debug_max_steps,
    )

    if not dist.is_initialized() or dist.get_rank() == 0:
        text_generation(model, config, device, max_samples=2, use_cache=True)
    
    if use_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
