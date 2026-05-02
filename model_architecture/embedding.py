import torch.nn as nn
from predictive_coding.pc_layer import PCLayer

class Embedding_Layer(nn.Module):
    """
    Embedding layer with word and positional embeddings, layer normalization, dropout, and a predictive coding layer.
    """
    def __init__(self, config):
        super(Embedding_Layer, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embeddings = nn.Embedding(config.block_size, config.n_embed)
        self.rms_norm = nn.RMSNorm(config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        
        self.pc_layer= PCLayer(
            T=config.T,
            lr=config.lr,
            inference_lr=config.inference_lr,
            energy_fn_name=config.internal_energy_fn_name,
            pc_optimizer=config.pc_optimizer,
            pc_beta1=config.pc_beta1,
            pc_beta2=config.pc_beta2,
            pc_eps=config.pc_eps,
            pc_weight_decay=config.pc_weight_decay,
            pc_update_clamp=config.pc_update_clamp,
        )
