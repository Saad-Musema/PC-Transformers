import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Tuple

from utils.pc_utils import (
    x_init,
    step_embed,
    step_linear,
    step_attn,
    finalize_step,
)
from predictive_coding.lateral_connc import LateralConnections

class PCLayer(nn.Module):
    """
    Predictive Coding Layer wrapper that manages iterative inference state and
    delegates computation to helper functions (step_embed, step_attn, step_linear).
    """
    def __init__(
        self,
        T: int,
        lr: float,
        inference_lr: float,
        update_bias: bool,
        energy_fn_name: str,
        num_heads: Optional[int] = None,
        n_embed: Optional[int] = None,
    ):
        super().__init__()
        self.rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.T = T
        self.local_lr = lr
        self.inference_lr = inference_lr
        self.update_bias = update_bias
        self.clamp_value = 3.0
        self.energy_fn_name = energy_fn_name 
        self.num_heads = num_heads
        self.n_embed = n_embed
        
        self.lateral_connections: Dict[str, LateralConnections] = {}
        
        self._x_cache: Dict[str, torch.Tensor] = {}
        self._mu_cache: Dict[str, torch.Tensor] = {}
        self._error_cache: Dict[str, torch.Tensor] = {}
        self._energy = 0.0
        self._errors = []
        self.debug_name = "pc_layer"
        self.layer_type = "unknown"
        self._debugger: Optional[Callable[[dict], None]] = None
    
    def register_lateral(self, layer_type: str, size: int):
        """Create and register lateral connections for layer_type."""
        if layer_type not in self.lateral_connections:
            self.lateral_connections[layer_type] = LateralConnections(size, self.local_lr, self.inference_lr)
            self.add_module(f"lateral_{layer_type}", self.lateral_connections[layer_type])

    def _reset_step_state(self) -> None:
        """Reset step-local accumulators, kept for future extension."""
        return
    
    def _get_cached_state(self, layer_type: str):
        return self._x_cache.get(layer_type, None)

    def set_debugger(self, debugger: Optional[Callable[[dict], None]]):
        """Attach an optional debug callback for step-level tracing."""
        self._debugger = debugger

    @staticmethod
    def _tensor_abs_mean(tensor: Optional[torch.Tensor]) -> Optional[float]:
        if tensor is None:
            return None
        return float(tensor.detach().abs().mean().item())

    @staticmethod
    def _parameter_snapshot(
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        stats: Dict[str, torch.Tensor] = {}

        if layer is not None and hasattr(layer, "weight") and layer.weight is not None:
            stats["weight"] = layer.weight.detach().clone()
            if getattr(layer, "bias", None) is not None:
                stats["bias"] = layer.bias.detach().clone()

        if proj_layers is not None:
            for name, module in proj_layers.items():
                if module is None or not hasattr(module, "weight") or module.weight is None:
                    continue
                stats[f"{name}.weight"] = module.weight.detach().clone()
                if getattr(module, "bias", None) is not None:
                    stats[f"{name}.bias"] = module.bias.detach().clone()

        return stats

    @staticmethod
    def _parameter_change_stats(
        before: Dict[str, torch.Tensor],
        after: Dict[str, torch.Tensor],
    ) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}

        for key in sorted(set(before) | set(after)):
            if key not in before or key not in after:
                continue

            before_tensor = before[key]
            after_tensor = after[key]
            delta = after_tensor - before_tensor
            stats[key] = {
                "before_abs_mean": float(before_tensor.abs().mean().item()),
                "after_abs_mean": float(after_tensor.abs().mean().item()),
                "delta_abs_mean": float(delta.abs().mean().item()),
                "delta_norm": float(delta.norm().item()),
            }

        return stats

    def _emit_debug(
        self,
        *,
        t: int,
        target_activity: torch.Tensor,
        mu: torch.Tensor,
        error: torch.Tensor,
        td_err: Optional[torch.Tensor],
        x_before: Optional[torch.Tensor],
        x_after: Optional[torch.Tensor],
        energy: float,
        param_changes: Dict[str, Dict[str, float]],
    ) -> None:
        if self._debugger is None:
            return

        self._debugger(
            {
                "layer_name": self.debug_name,
                "layer_type": self.layer_type,
                "pc_step": t,
                "target_abs_mean": self._tensor_abs_mean(target_activity),
                "mu_abs_mean": self._tensor_abs_mean(mu),
                "error_abs_mean": self._tensor_abs_mean(error),
                "td_err_abs_mean": self._tensor_abs_mean(td_err),
                "x_before_abs_mean": self._tensor_abs_mean(x_before),
                "x_after_abs_mean": self._tensor_abs_mean(x_after),
                "x_delta_abs_mean": self._tensor_abs_mean(
                    None if x_before is None or x_after is None else x_after - x_before
                ),
                "energy": energy,
                "param_changes": param_changes,
            }
        )
    
    def forward(
        self,
        target_activity: torch.Tensor,
        layer_type: str,
        t: int,
        T: int,
        requires_update: bool,
        td_err:  Optional[torch.Tensor] = None,
        layer: Optional[nn.Module] = None,
        layer_norm: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        flash: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False, 
        output_proj: Optional[nn.Module] = None,
        attn_lr_multiplier: float = 1.0,
    ):
        """Perform one predictive coding inference step."""
        self._reset_step_state()
        x = self._get_cached_state(layer_type)
        x_before = x.detach().clone() if isinstance(x, torch.Tensor) and self._debugger is not None else None
        params_before = self._parameter_snapshot(layer=layer, proj_layers=proj_layers) if self._debugger is not None else {}

        if rope_cache is not None:
            self.rope_cache = rope_cache


        if layer_type == "embed":
            mu, mu_word, bu_err = step_embed(
                t,
                T,
                target_activity,
                layer,
                layer_type,
                input_ids,
                self.local_lr,
                self.clamp_value,
                self.energy_fn_name,
                requires_update,
                layer_norm=layer_norm,
            )            
            # store for later retrieval
            self._x_cache["embed"] = (mu_word)
            self._mu_cache["embed"] = mu.detach().clone()
            if bu_err is not None:
                self._error_cache["embed"] = bu_err.detach().clone()

            # compute energy
            error = target_activity - mu
            energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
            self._energy += energy
            self._errors.extend(step_errors)
            self._emit_debug(
                t=t,
                target_activity=target_activity,
                mu=mu,
                error=error,
                td_err=td_err,
                x_before=None,
                x_after=None,
                energy=energy,
                param_changes=self._parameter_change_stats(
                    params_before,
                    self._parameter_snapshot(layer=None, proj_layers=layer),
                ),
            )
            return mu_word
        
        elif layer_type == "attn":
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err, new_kv_cache = step_attn(
                t,
                T,
                target_activity,
                x,
                lateral_conn,
                proj_layers,
                layer_type,
                self.local_lr,
                self.inference_lr,
                self.clamp_value,
                self.energy_fn_name,
                self.update_bias,
                requires_update,
                self.num_heads,
                self.n_embed,
                td_err=td_err, 
                layer_norm=layer_norm,
                rope_cache=self.rope_cache, 
                flash=flash, 
                kv_cache=kv_cache,  
                use_cache=use_cache,
                output_proj=output_proj,
                attn_lr_multiplier=attn_lr_multiplier,
            )
            # Store cache for retrieval
            if use_cache:
                self._last_kv_cache = new_kv_cache
        
        else:
            lateral_conn = self.lateral_connections.get(layer_type, None)
            x, mu, bu_err = step_linear(
                t,
                T,
                target_activity,
                x,
                layer, 
                lateral_conn,  
                layer_type,
                self.local_lr, 
                self.inference_lr,
                self.clamp_value, 
                self.energy_fn_name, 
                self.update_bias, 
                requires_update,
                td_err=td_err, 
                layer_norm=layer_norm
            )
            
        # cache and stats
        self._mu_cache[layer_type] = mu.detach().clone()  
        if bu_err is not None: 
         self._error_cache[layer_type] = bu_err.detach().clone()   
        
        error = target_activity - mu
        energy, step_errors = finalize_step(mu, target_activity, error, t, layer_type, self.energy_fn_name)
        self._energy += energy
        self._errors.extend(step_errors)

        # update x cache
        self._x_cache[layer_type] = x
        self._emit_debug(
            t=t,
            target_activity=target_activity,
            mu=mu,
            error=error,
            td_err=td_err,
            x_before=x_before,
            x_after=x,
            energy=energy,
            param_changes=self._parameter_change_stats(
                params_before,
                self._parameter_snapshot(layer=layer, proj_layers=proj_layers),
            ),
        )
        return x, mu

    def init_x(
        self,
        batch_size: int,
        seq_len: int,
        layer_type: str,
        device: torch.device,
        layer: Optional[nn.Module] = None,
        proj_layers: Optional[dict] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Initialize cached activity `x` for the layer type.
        - embed: stores (x_word, x_pos) from embedding weights
        - attn: creates random initialization shaped (B, S, H_out)
        - linear/others: random init sized to layer input dimension
        """
        if layer_type == "embed":
            assert input_ids is not None and position_ids is not None, "Embedding layer requires input_ids and position_ids"
            vocab_size = layer["word"].weight.size(0)
            if input_ids.max() >= vocab_size:
                input_ids = torch.clamp(input_ids, max=vocab_size-1)
            
            max_pos = layer["pos"].weight.size(0)
            if position_ids.max() >= max_pos:
                position_ids = torch.clamp(position_ids, max=max_pos-1)
            
            x_word = layer["word"].weight[input_ids] 
            x_pos = layer["pos"].weight[position_ids] 
            self._x_cache["embed"] = (x_word, x_pos)
            
        elif layer_type == "attn":
            assert proj_layers is not None, "Attention layer requires proj_layers"
            H_in = proj_layers["q_proj"].weight.shape[1]
            H_out = proj_layers["v_proj"].weight.shape[0] 
            self._x_cache["attn"] = x_init(batch_size, seq_len, H_out, device)
            
            self.register_lateral(layer_type, H_in)
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
        
        else:  
            assert layer is not None, "Linear layer requires layer parameter"
            input_dim = layer.weight.shape[1]
            self._x_cache[layer_type] = x_init(batch_size, seq_len, input_dim, device)
            
            self.register_lateral(layer_type, input_dim)  
            if layer_type in self.lateral_connections:
                self.lateral_connections[layer_type] = self.lateral_connections[layer_type].to(device) 
    
    def get_x(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached activity tensor for a given layer type."""
        return self._x_cache.get(layer_type, None)
    
    def get_mu(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached mu (prediction) tensor for a given layer type."""
        return self._mu_cache.get(layer_type, None)
    
    def get_td_err(self, layer_type: str) -> Optional[torch.Tensor]:
        """Get the cached top-down error tensor for a given layer type."""
        return self._error_cache.get(layer_type, None)

    def get_energy(self) -> Optional[float]:
        """Get the accumulated energy for the layer."""
        return float(self._energy)

    def clear_energy(self):
        """Clear the stored energy and cached states for the layer."""
        self._energy = 0.0
        self._x_cache.clear()
        self._mu_cache.clear()
        
    def get_errors(self) -> list:
        """Get the list of error values accumulated during inference."""
        return self._errors

    def clear_errors(self):
        """Clear the stored errors for the layer."""
        self._errors = []
        self._error_cache.clear()
        
    def set_learning_rate(self, lr: float):
        """Set the local learning rate for the layer."""
        self.local_lr = float(lr)

        for lateral_conn in self.lateral_connections.values():
            lateral_conn.set_learning_rate(self.local_lr)

    def set_inference_learning_rate(self, inference_lr: float):
        """Set the inference learning rate for the layer."""
        self.inference_lr = float(inference_lr)

        for lateral_conn in self.lateral_connections.values():
            lateral_conn.set_inference_learning_rate(self.inference_lr)
        
    def get_learning_rate(self) -> float:
        """Get the current local learning rate for the layer."""
        return float(self.local_lr)
    
    def get_inference_learning_rate(self) -> float:
        """Get the current inference learning rate for the layer."""
        return float(self.inference_lr)
