import logging
from predictive_coding.config import GPTConfig, average_layer_learning_rate, build_pc_layer_names

logger = logging.getLogger(__name__)


def _trial_lr_name(layer_name: str) -> str:
    return f"peak_lr__{layer_name.replace('.', '__')}"

def get_dynamic_model_config(trial, vocab_size, flash=False):
    """Get model configuration with dynamic parameter combinations, including flash attention flag."""
    n_embed = trial.suggest_int("n_embed", 64, 512, step=16)

    valid_heads = [h for h in range(2, min(32, n_embed // 8) + 1) if n_embed % h == 0 and 8 <= n_embed // h <= 128]
    if not valid_heads:
        logger.warning(f"No valid heads for n_embed={n_embed}, forcing fallback.")
        return None
        
    num_heads = valid_heads[trial.suggest_int('head_idx', 0, len(valid_heads) - 1)]
    block_size = trial.suggest_int("block_size", 64, 512, step=16)
    n_blocks = trial.suggest_int('n_blocks', 1, 12)
    T = trial.suggest_int('T', 1, 14, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    layer_peak_lrs = {
        layer_name: trial.suggest_float(_trial_lr_name(layer_name), 1e-5, 1e-2, log=True)
        for layer_name in build_pc_layer_names(n_blocks)
    }
    layer_lrs = {layer_name: peak_lr * 0.1 for layer_name, peak_lr in layer_peak_lrs.items()}
    lr = average_layer_learning_rate(layer_lrs)
    peak_lr = average_layer_learning_rate(layer_peak_lrs)
    inference_lr = lr * 100
    warmup_steps = trial.suggest_int('warmup_steps', 50, 2000, log=True)
    update_bias = trial.suggest_int('update_bias_int', 0, 1) == 1
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    combined_internal_weight = trial.suggest_float('combined_internal_weight', 0.1, 0.9)
    combined_output_weight = 1.0 - combined_internal_weight
    num_epochs = num_epochs = 10
    alpha = 0.5
    
    return GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        peak_learning_rate=peak_lr,
        warmup_steps=warmup_steps,
        n_embed=n_embed,
        dropout=dropout,
        lr=lr,
        inference_lr=inference_lr,
        T=T,
        num_heads=num_heads,
        n_blocks=n_blocks,
        batch_size = batch_size,
        num_epochs=num_epochs,
        update_bias=update_bias,
        internal_energy_fn_name="pc_e",
        output_energy_fn_name="pc_e",
        combined_internal_weight = combined_internal_weight,
        combined_output_weight = combined_output_weight,
        use_flash_attention=flash,
        alpha=alpha,
        layer_lrs=layer_lrs,
        layer_peak_lrs=layer_peak_lrs,
    )

def update_global_config(config):
    """Update global GPTConfig"""
    config_keys = [
        'num_heads', 'n_embed', 'block_size', 'n_blocks', 'vocab_size',
        'dropout', 'lr', 'inference_lr', 'peak_learning_rate', 'warmup_steps',
        'update_bias', 'T', 'internal_energy_fn_name', 'output_energy_fn_name',
        'batch_size', 'num_epochs', 'combined_internal_weight', 
        'combined_output_weight', 'alpha', 'layer_lrs', 'layer_peak_lrs'
    ]
    
    for key in config_keys:
        try:
            if isinstance(config, dict):
                if key in config:
                    setattr(GPTConfig, key, config[key])
            elif hasattr(config, key):
                setattr(GPTConfig, key, getattr(config, key))
        except Exception as e:
            logger.warning(f"Failed to update config key '{key}': {e}")
            continue
