from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional


def build_pc_layer_names(n_blocks: int) -> list[str]:
    """Return the canonical predictive-coding layer names for a model depth."""
    layer_names = ["embedding"]

    for idx in range(n_blocks):
        layer_names.extend(
            [
                f"block_{idx}.attn.qkv",
                f"block_{idx}.attn.output",
                f"block_{idx}.mlp.fc1",
                f"block_{idx}.mlp.fc2",
            ]
        )

    layer_names.append("output")
    return layer_names


def average_layer_learning_rate(layer_lrs: Mapping[str, float]) -> float:
    """Return the mean LR from a per-layer learning-rate mapping."""
    if not layer_lrs:
        raise ValueError("layer_lrs must not be empty")
    return float(sum(float(lr) for lr in layer_lrs.values()) / len(layer_lrs))


def build_layer_lr_map(
    n_blocks: int,
    layer_lrs: Optional[Mapping[str, float]] = None,
    *,
    fallback: Optional[float] = None,
    fallback_map: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """
    Build a complete canonical LR map for the current model depth.

    Extra keys are ignored so configs can be reused across different `n_blocks`
    values. Missing keys are filled from `fallback_map` and then `fallback`.
    """
    layer_names = build_pc_layer_names(n_blocks)
    resolved: Dict[str, float] = {}

    if layer_lrs is not None:
        for name, value in layer_lrs.items():
            if name in layer_names:
                resolved[name] = float(value)

    for name in layer_names:
        if name in resolved:
            continue
        if fallback_map is not None and name in fallback_map:
            resolved[name] = float(fallback_map[name])
            continue
        if fallback is not None:
            resolved[name] = float(fallback)
            continue
        raise ValueError(f"Missing learning rate for layer '{name}'")

    return resolved


def resolve_layer_learning_rates(
    *,
    n_blocks: int,
    layer_lrs: Optional[Mapping[str, float]] = None,
    layer_peak_lrs: Optional[Mapping[str, float]] = None,
    lr: Optional[float] = None,
    peak_learning_rate: Optional[float] = None,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Resolve canonical base and peak LR maps from layered or legacy fields."""
    initial_peak_lrs = None
    if layer_peak_lrs is not None or peak_learning_rate is not None:
        initial_peak_lrs = build_layer_lr_map(
            n_blocks,
            layer_peak_lrs,
            fallback=peak_learning_rate,
        )

    layer_lrs_resolved = build_layer_lr_map(
        n_blocks,
        layer_lrs,
        fallback=lr,
        fallback_map=(
            {name: peak_lr * 0.1 for name, peak_lr in initial_peak_lrs.items()}
            if initial_peak_lrs is not None
            else None
        ),
    )

    layer_peak_lrs_resolved = build_layer_lr_map(
        n_blocks,
        layer_peak_lrs,
        fallback=peak_learning_rate,
        fallback_map=None if peak_learning_rate is not None else layer_lrs_resolved,
    )

    return layer_lrs_resolved, layer_peak_lrs_resolved

@dataclass
class GPTConfig:
    """
    Configuration dataclass for the predictive coding transformer model.

    Attributes:
        vocab_size (int): Size of the vocabulary.
        block_size (int): Maximum sequence length.
        n_embed (int): Embedding dimension size.
        dropout (float): Dropout probability.
        layer_lrs (dict[str, float]): Base local learning rates keyed by layer name.
        layer_peak_lrs (dict[str, float]): Peak scheduled learning rates keyed by layer name.
        warmup_steps (int): Number of warmup steps for learning rate scheduling.
        T (int): Number of inference steps for predictive coding.
        update_bias (bool): Whether to update bias terms during learning.
        num_heads (int): Number of attention heads.
        n_blocks (int): Number of transformer blocks.
        batch_size (int): Batch size for training/evaluation.
        num_epochs (int): Number of training epochs.
        energy_fn_name (str): Name of the energy function to use for error computation.
        use_flash_attention (bool): Whether to use FlashAttention.
    """
    vocab_size: int
    block_size: int
    inference_lr: float
    warmup_steps: Optional[int] 
    n_embed: int 
    dropout: float 
    T: int 
    update_bias: bool 
    num_heads: int 
    n_blocks: int 
    batch_size: int
    num_epochs: int
    internal_energy_fn_name:str
    output_energy_fn_name: str
    combined_internal_weight: float 
    combined_output_weight: float
    use_flash_attention: bool
    alpha: float
    layer_lrs: Dict[str, float] = field(default_factory=dict)
    layer_peak_lrs: Dict[str, float] = field(default_factory=dict)
    lr: Optional[float] = None
    peak_learning_rate: Optional[float] = None

    def __post_init__(self) -> None:
        self.layer_lrs, self.layer_peak_lrs = resolve_layer_learning_rates(
            n_blocks=self.n_blocks,
            layer_lrs=self.layer_lrs,
            layer_peak_lrs=self.layer_peak_lrs,
            lr=self.lr,
            peak_learning_rate=self.peak_learning_rate,
        )
        self.lr = float(self.lr) if self.lr is not None else average_layer_learning_rate(self.layer_lrs)
        self.peak_learning_rate = (
            float(self.peak_learning_rate)
            if self.peak_learning_rate is not None
            else average_layer_learning_rate(self.layer_peak_lrs)
        )
