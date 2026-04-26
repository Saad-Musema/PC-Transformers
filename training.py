import torch
import os
import math
import time
import torch.nn.functional as F
import torch.distributed as dist
import argparse
import json
import logging
from predictive_coding.config import GPTConfig
from predictive_coding.pc_layer import PCLayer
from model_architecture.pc_t_model import PCTransformer
from data_preparation.dataloader import get_loaders
from utils.config_utils import load_best_config
from utils.model_utils import set_seed
from eval import evaluate
from visualization import plot_metrics
from utils.device_utils import setup_device, synchronize_model_parameters
from data_preparation.config import vocab_size

"""
This script trains the predictive coding transformer model on the provided dataset.
It tracks and plots the average predictive coding energy per epoch and saves the trained model.

Usage: torchrun --nproc-per-node=<NUM_GPU> training.py

"""

def _get_scheduled_layer_lrs(config, global_step):
    """Return the current per-layer LR schedule value for this optimization step."""
    if config.warmup_steps and global_step < config.warmup_steps:
        warmup_progress = global_step / max(config.warmup_steps, 1)
        return {
            layer_name: base_lr + warmup_progress * (config.layer_peak_lrs[layer_name] - base_lr)
            for layer_name, base_lr in config.layer_lrs.items()
        }

    return dict(config.layer_peak_lrs)

def train(model, dataloader, config, global_step, device, logger):
    model.train()
    total_ce_loss = 0.0
    total_energy = 0.0
    batch_count = 0

    base_model = model.module if hasattr(model, 'module') else model
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        # total_steps = len(dataloader) * config.num_epochs
        
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size - 1)

        scheduled_layer_lrs = _get_scheduled_layer_lrs(config, global_step)
        if hasattr(base_model, "set_layer_learning_rates"):
            base_model.set_layer_learning_rates(scheduled_layer_lrs)
        else:
            for module in model.modules():
                layer_name = getattr(module, "debug_name", None)
                if hasattr(module, "local_lr") and layer_name in scheduled_layer_lrs:
                    module.set_learning_rate(scheduled_layer_lrs[layer_name])

        global_step += 1
        if target_ids.max() >= vocab_size:
            target_ids = torch.clamp(target_ids, max=vocab_size-1)

        if hasattr(base_model, "set_debug_context"):
            base_model.set_debug_context(
                mode="train",
                batch_idx=batch_idx,
                global_step=global_step,
            )

        logits = model(target_ids, input_ids)
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=0
        )
        total_ce_loss += ce_loss.item()

        internal_energies = []
        output_energy = None

        for module in model.modules():
            if isinstance(module, PCLayer) and hasattr(module, "get_energy"):
                energy = module.get_energy()
                if energy is None or (isinstance(energy, float) and math.isnan(energy)):
                    continue

                if hasattr(module, 'layer_type') and module.layer_type == 'linear_output':
                    if getattr(module, 'energy_fn_name', None) == "kld":
                        output_energy = energy
                    else:
                        internal_energies.append(energy)
                else:
                    internal_energies.append(energy)

                if hasattr(module, "_head_similarity_avg"):
                    _ = module._head_similarity_avg
                if hasattr(module, "_head_similarity_max"):
                    _ = module._head_similarity_max

        synchronize_model_parameters(base_model)

        avg_internal_energy = sum(internal_energies) / len(internal_energies) if internal_energies else ce_loss.item()
                
        if output_energy is not None:
            batch_energy = config.combined_internal_weight * avg_internal_energy + config.combined_output_weight * output_energy 
        else:
            batch_energy = avg_internal_energy
        total_energy += batch_energy
        batch_count += 1

        perplexity = math.exp(ce_loss.item()) if ce_loss.item() < 100 else float("inf")

        if (not dist.is_initialized() or dist.get_rank() == 0) and (batch_idx + 1) % 10 == 0:
            if logger:
                logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}")
            else:
                print(f"  Batch {batch_idx + 1}/{len(dataloader)} | Batch Energy: {batch_energy:.4f} | Perplexity: {perplexity:.4f}")

    if dist.is_initialized():
        stats = torch.tensor([total_energy, total_ce_loss, batch_count], device=device, dtype=torch.float64)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        total_energy, total_ce_loss, batch_count = stats.tolist()

    avg_energy = total_energy / batch_count if batch_count > 0 else 0.0
    avg_ce_loss = total_ce_loss / batch_count if batch_count > 0 else 0.0
    avg_perplexity = math.exp(avg_ce_loss) if avg_ce_loss < 100 else float("inf")
    return avg_energy, avg_perplexity, global_step


def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description="Train the predictive coding transformer")
    parser.add_argument('--debug-layers', action='store_true', help='Log per-layer predictive-coding stats')
    parser.add_argument('--debug-max-batches', type=int, default=1, help='Number of batches to trace when layer debug is enabled')
    parser.add_argument('--debug-max-steps', type=int, default=None, help='Maximum predictive-coding iterations to trace')
    parser.add_argument('--debug-rank', type=int, default=0, help='Distributed rank that emits layer debug logs')
    args = parser.parse_args()

    local_rank, device, use_distributed = setup_device()
    if use_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    rank = dist.get_rank() if dist.is_initialized() else 0

    best_config = load_best_config()   
    # Configure logging
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # build handlers and remove existing ones
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_h = logging.StreamHandler()
    stream_h.setFormatter(fmt)
    root_logger.addHandler(stream_h)

    if rank == 0:
        file_h = logging.FileHandler(os.path.join(log_dir, "training.log"), mode="a")
        file_h.setFormatter(fmt)
        root_logger.addHandler(file_h)

    logger = logging.getLogger(__name__)
   
    config = GPTConfig(
        vocab_size = vocab_size,
        block_size = best_config["block_size"],
        lr = best_config["lr"],
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
        use_flash_attention=best_config["use_flash_attention"],
        alpha = best_config["alpha"],
        layer_lrs=best_config["layer_lrs"],
        layer_peak_lrs=best_config["layer_peak_lrs"],
    )
    
    # Create a separate logger for hyperparameters
    param_logger = logging.getLogger('param_logger')
    param_logger.setLevel(logging.INFO)
    if rank == 0 and root_logger.handlers:
        param_logger.addHandler(root_logger.handlers[1])
        param_logger.propagate = False

    if rank == 0:
        param_logger.info(f"\n{'#' * 120}") 
        logger.info(f"Using device: {device} (local rank {local_rank})")
        try:
            cfg = config.__dict__
        except Exception:
            cfg = {k: getattr(config, k) for k in dir(config) if not k.startswith("_") and not callable(getattr(config, k))}
        config_json = json.dumps(cfg, indent=6, default=str)
        param_logger.info("Saving the hyperparameters configurations:")
        param_logger.info(config_json)

    model = PCTransformer(config).to(device)
    model.configure_layer_debug(
        enabled=args.debug_layers and rank == args.debug_rank,
        log_fn=logger.info,
        max_batches=args.debug_max_batches,
        max_steps=args.debug_max_steps,
    )

    train_loader, valid_loader, _ = get_loaders(
        distributed=use_distributed,
        batch_size=config.batch_size,
    )
    
    global_step = 0
    train_energies = []
    val_energies = []
    train_perplexities = [] 
    val_perplexities = []

    start_time = time.time()
    if rank == 0:
        logger.info("========== Training started ==========") 
        logger.info(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
        if use_distributed:
            logger.info("Distributed training enabled with manual parameter averaging after each batch")

    for epoch in range(config.num_epochs):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, torch.utils.data.DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        model.train()
        train_energy, train_perplexity, global_step = train(
            model, train_loader, config, global_step, device, logger
        )
        train_energies.append(train_energy)
        train_perplexities.append(train_perplexity)

        model.eval()
        with torch.no_grad():
            val_energy, val_perplexity = evaluate(
                model, config, valid_loader, max_batches=None, device=device
            )
        
        val_energies.append(val_energy)
        val_perplexities.append(val_perplexity)

        model.train()

        if rank == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs} | "
                  f"Train Energy: {train_energy:.4f} | Train Perplexity: {train_perplexity:.4f} | "
                  f"Val Energy: {val_energy:.4f} | Val Perplexity: {val_perplexity:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == config.num_epochs - 1:
                os.makedirs("checkpoints", exist_ok=True)
                # Get the underlying model (handle both DDP and non-DDP cases)
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint = {
                    'model_state_dict': model_to_save.state_dict(),
                }
                checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    if rank == 0:
        plot_metrics(
            train_energies,
            val_energies,
            train_perplexities,
            val_perplexities
        )

        os.makedirs("checkpoints", exist_ok=True)
        # Get the underlying model (handle both DDP and non-DDP cases)
        model_to_save = model.module if hasattr(model, 'module') else model
        final_checkpoint = {
            'epoch': config.num_epochs,
            'model_state_dict': model_to_save.state_dict(),
            'train_energy': train_energy,
            'val_energy': val_energy,
            'train_perplexity': train_perplexity,
            'val_perplexity': val_perplexity
        }
        torch.save(final_checkpoint, 'checkpoints/final_model.pt')
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info("Final model saved to: checkpoints/final_model.pt")
        logger.info("========== Training completed ==========")

    # dist.destroy_process_group()
    if use_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
