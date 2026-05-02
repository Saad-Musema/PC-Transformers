import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class OutputLayer(nn.Module):
    """
    Output layer for the transformer model, consisting of a linear projection and a predictive coding layer.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output = nn.Linear(config.n_embed, config.vocab_size)
        
        self.pc_layer = PCLayer(
            T=config.T,
            lr=config.lr,
            inference_lr=config.inference_lr,
            energy_fn_name=config.output_energy_fn_name,
            pc_optimizer=config.pc_optimizer,
            pc_beta1=config.pc_beta1,
            pc_beta2=config.pc_beta2,
            pc_eps=config.pc_eps,
            pc_weight_decay=config.pc_weight_decay,
            pc_update_clamp=config.pc_update_clamp,
        )
