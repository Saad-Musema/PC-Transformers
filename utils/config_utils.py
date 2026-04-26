import os
import re
import ast
import json

from predictive_coding.config import average_layer_learning_rate, resolve_layer_learning_rates


def _parse_config_value(value: str):
    value = value.strip()

    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(value)
        except (json.JSONDecodeError, ValueError, SyntaxError):
            continue

    if value.lower() in {"true", "false"}:
        return value.lower() == "true"

    try:
        num = float(value)
        return int(num) if num.is_integer() else num
    except ValueError:
        return value.strip('"').strip("'")

def load_best_config():
    """
    Parses a result file and returns a dict of selected hyperparameters.
    If the file is missing or a key is missing, fallback values are used.
    """

    selected_keys = {
        "block_size", "peak_learning_rate", "warmup_steps", "n_embed",
        "dropout", "T", "num_heads", "n_blocks", "update_bias", "alpha",
        "lr", "inference_lr", "batch_size", "num_epochs", "internal_energy_fn_name",
        "output_energy_fn_name", "combined_internal_weight",
        "combined_output_weight", "use_flash_attention", "layer_lrs", "layer_peak_lrs"
    }

    fallback_values = {
        "block_size": 64,
        "peak_learning_rate": 0.009606017304857476,
        "warmup_steps": 59,
        "n_embed": 512,
        "dropout": 0.15,
        "T": 10,
        "num_heads": 32,
        "n_blocks": 12,
        "update_bias": False,
        "alpha": 0.5,
        "lr": 0.0009606017304857476,
        "inference_lr": 0.096,
        "batch_size": 8,
        "num_epochs": 10,
        "internal_energy_fn_name": "pc_e",
        "output_energy_fn_name": "pc_e",
        "combined_internal_weight": 0.8779955579743048,
        "combined_output_weight": 0.12200444202569516,
        "use_flash_attention": False,
        "layer_lrs": None,
        "layer_peak_lrs": None,
    }

    config = {}
    file_path = os.path.join(os.path.dirname(__file__), "..", "tuning", "bayesian_tuning_results.txt")

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()

        for line in content.splitlines():
            match = re.match(r'(\w+):\s+(.*)', line)
            if match:
                key, value = match.groups()
                if key in selected_keys:
                    config[key] = _parse_config_value(value)
    else:
        print(f"[WARNING] Tuning result file not found: {file_path}")
        print(f"[INFO] Using fallback values for missing keys: {selected_keys - config.keys()}")

    # Fill in missing keys from fallback
    for key in selected_keys:
        if key not in config:
            config[key] = fallback_values[key]

    if config["inference_lr"] is None and config["lr"] is not None:
        config["inference_lr"] = float(config["lr"]) * 100.0

    layer_lrs, layer_peak_lrs = resolve_layer_learning_rates(
        n_blocks=int(config["n_blocks"]),
        layer_lrs=config.get("layer_lrs"),
        layer_peak_lrs=config.get("layer_peak_lrs"),
        lr=config.get("lr"),
        peak_learning_rate=config.get("peak_learning_rate"),
    )

    config["layer_lrs"] = layer_lrs
    config["layer_peak_lrs"] = layer_peak_lrs
    config["lr"] = float(config["lr"]) if config["lr"] is not None else average_layer_learning_rate(layer_lrs)
    config["peak_learning_rate"] = (
        float(config["peak_learning_rate"])
        if config["peak_learning_rate"] is not None
        else average_layer_learning_rate(layer_peak_lrs)
    )

    return config
