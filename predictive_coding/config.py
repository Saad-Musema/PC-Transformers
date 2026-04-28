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


def build_layer_lr_map(
    n_blocks: int,
    layer_lrs: Dict[str, float],
    base_lr: float,
) -> Dict[str, float]:
    """Build canonical LR map from simple group LRs."""
    layer_names = build_pc_layer_names(n_blocks)
    resolved: Dict[str, float] = {}
    
    for name in layer_names:
        if name == "embedding":
            resolved[name] = layer_lrs.get("embedding", base_lr)
        elif name == "output":
            resolved[name] = layer_lrs.get("output", base_lr)
        elif "attn" in name:
            resolved[name] = layer_lrs.get("attention", base_lr)
        elif "mlp" in name:
            resolved[name] = layer_lrs.get("mlp", base_lr)
    
    return resolved


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
        if self.layer_lrs and self.layer_peak_lrs:
            return
        
        base_lr = self.lr or 1e-4
        base_peak = self.peak_learning_rate or base_lr
        
        simple_layer_lrs = {
            "embedding": base_lr,
            "attention": base_lr,
            "mlp": base_lr,
            "output": base_lr,
        }
        simple_peak_lrs = {
            "embedding": base_peak,
            "attention": base_peak,
            "mlp": base_peak,
            "output": base_peak,
        }
        
        if self.layer_lrs:
            for k, v in self.layer_lrs.items():
                if k in simple_layer_lrs:
                    simple_layer_lrs[k] = v
            if self.layer_peak_lrs:
                for k, v in self.layer_peak_lrs.items():
                    if k in simple_peak_lrs:
                        simple_peak_lrs[k] = v
        
        self.layer_lrs = build_layer_lr_map(self.n_blocks, simple_layer_lrs, base_lr)
        self.layer_peak_lrs = build_layer_lr_map(self.n_blocks, simple_peak_lrs, base_peak)
        
        self.lr = base_lr
        self.peak_learning_rate = base_peak