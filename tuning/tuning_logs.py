import logging
import os
import json

def initialize_logs(study_name: str):
    """Create and initialize summary and trial log files."""
    trials_path = f"tuning/{study_name}_trials.txt"


    with open(trials_path, "w") as f:
        f.write(f"DETAILED TRIAL RESULTS - {study_name}\n")
        f.write(f"{'='*50}\n")
        f.write("Objective: Minimize Averge Energy \n\n")

    return trials_path
def log_trial_to_detailed_log(trials_path, trial, config, trial_time, avg_energy, write_header=False):
    """Append trial information as a readable structured block."""
    avg_perplexity = trial.user_attrs.get("perplexity", "N/A")
    combined_loss = trial.user_attrs.get("combined_loss", "N/A")
    
    with open(trials_path, "a") as f:
        if write_header:
            f.write("Each trial stores summary metrics plus full per-layer LR maps.\n")
            f.write("-" * 120 + "\n")

        trial_record = {
            "trial": trial.number,
            "time_s": round(trial_time, 1),
            "avg_energy": avg_energy,
            "perplexity": avg_perplexity,
            "combined_loss": combined_loss,
            "n_embed": config.n_embed,
            "block_size": config.block_size,
            "num_heads": config.num_heads,
            "n_blocks": config.n_blocks,
            "T": config.T,
            "lr_avg": config.lr,
            "peak_lr_avg": config.peak_learning_rate,
            "inference_lr": config.inference_lr,
            "warmup_steps": config.warmup_steps,
            "dropout": config.dropout,
            "update_bias": config.update_bias,
            "layer_lrs": config.layer_lrs,
            "layer_peak_lrs": config.layer_peak_lrs,
        }

        for key, value in trial_record.items():
            serialized = json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            f.write(f"{key}: {serialized}\n")
        f.write("-" * 120 + "\n")
        
def write_final_results(results_path, trial):
    config = trial.user_attrs.get("config", {})
    energy = trial.user_attrs.get("energy", "N/A")
    perplexity = trial.user_attrs.get("perplexity", "N/A")
    combined_loss = trial.user_attrs.get("combined_loss", "N/A")
    
    with open(results_path, "w") as f:
        f.write("COMBINED ENERGY OPTIMIZATION RESULTS\n")
        f.write("====================================\n\n")
        f.write(f"Best combined energy: {trial.value:.4f}\n")
        f.write(f"Average Energy: {energy:.4f}\n")
        f.write(f"Average Perplexity: {perplexity:.4f}\n")
        f.write(f"Combined Loss: {combined_loss:.4f}\n\n")
        
        if config:
            f.write("Best Configuration:\n")
            for key, val in config.items():
                serialized = json.dumps(val, sort_keys=True) if isinstance(val, dict) else val
                f.write(f"{key}: {serialized}\n")

def trial_batch_logger(trial_number: int, log_dir: str = "logs") -> logging.Logger:
    """
    Returns a logger that prepends trial and epoch info to every message.
    
    Args:
        trial_number: Current trial number
        log_dir: Directory where batch logs will be saved
    """
    os.makedirs(log_dir, exist_ok=True)
    
    base_logger = logging.getLogger(f"trial_{trial_number}")
    base_logger.setLevel(logging.INFO)
    
    if not base_logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "batch_debug.log"), mode="a")
        fmt = logging.Formatter(f"[Trial {trial_number} | %(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(fmt)
        base_logger.addHandler(fh)
    
    return base_logger
