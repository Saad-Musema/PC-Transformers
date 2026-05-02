import torch


class PCOptimizer:
    """Applies optimizer formulas to precomputed predictive-coding updates."""

    def __init__(
        self,
        method: str = "adam_mini",
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        update_clamp: float = 0.01,
    ):
        if method not in {"sgd", "adam_mini"}:
            raise ValueError(f"Unknown PC optimizer: {method}")
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.update_clamp = update_clamp
        self.state = {}

    def step(self, param: torch.nn.Parameter, update: torch.Tensor, lr: float, block: str = "full") -> None:
        """Apply a precomputed PC update direction to a parameter tensor."""
        if self.method == "sgd":
            applied = lr * update
        else:
            applied = lr * self._adam_mini_update(param, update, block)

        applied = torch.clamp(applied, -abs(self.update_clamp), abs(self.update_clamp))
        if self.weight_decay:
            param.data.mul_(1.0 - lr * self.weight_decay)
        param.data.add_(applied)

    def index_step(self, param: torch.nn.Parameter, indices: torch.Tensor, update: torch.Tensor, lr: float) -> None:
        """Apply row-wise PC updates for embedding/token blocks."""
        if self.method == "sgd":
            applied = lr * update
            applied = torch.clamp(applied, -abs(self.update_clamp), abs(self.update_clamp))
            param.data.index_add_(0, indices, applied)
        else:
            unique_indices, inverse = torch.unique(indices, sorted=False, return_inverse=True)
            row_update = torch.zeros(
                unique_indices.size(0),
                update.size(1),
                device=update.device,
                dtype=update.dtype,
            )
            row_update.index_add_(0, inverse, update)
            applied = lr * self._adam_mini_index_update(param, unique_indices, row_update)
            applied = torch.clamp(applied, -abs(self.update_clamp), abs(self.update_clamp))
            param.data.index_add_(0, unique_indices, applied)

    def state_numel(self) -> int:
        """Return total tensor elements held in optimizer state."""
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel()
        return total

    def state_bytes(self) -> int:
        """Return total tensor bytes held in optimizer state."""
        total = 0
        for state in self.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    total += value.numel() * value.element_size()
        return total

    def _adam_mini_update(self, param: torch.nn.Parameter, update: torch.Tensor, block: str) -> torch.Tensor:
        state = self.state.setdefault(id(param), {})
        if "m" not in state:
            state["m"] = torch.zeros_like(param.data)
            state["step"] = 0
        state["step"] += 1

        m = state["m"]
        m.mul_(self.beta1).add_(update, alpha=1.0 - self.beta1)
        v = self._update_block_v(state, update, block)

        m_hat = m / (1.0 - self.beta1 ** state["step"])
        v_hat = v / (1.0 - self.beta2 ** state["step"])
        return m_hat / (torch.sqrt(v_hat) + self.eps)

    def _adam_mini_index_update(self, param: torch.nn.Parameter, indices: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        state = self.state.setdefault(id(param), {})
        if "m" not in state:
            state["m"] = torch.zeros_like(param.data)
            state["v_rows"] = torch.zeros(param.data.size(0), device=param.data.device, dtype=param.data.dtype)
            state["step"] = 0
        state["step"] += 1

        m_rows = state["m"].index_select(0, indices)
        m_rows.mul_(self.beta1).add_(update, alpha=1.0 - self.beta1)
        state["m"].index_copy_(0, indices, m_rows)

        row_v = update.pow(2).mean(dim=1)
        old_v = state["v_rows"].index_select(0, indices)
        new_v = old_v.mul(self.beta2).add(row_v, alpha=1.0 - self.beta2)
        state["v_rows"].index_copy_(0, indices, new_v)

        m_hat = m_rows / (1.0 - self.beta1 ** state["step"])
        v_hat = new_v / (1.0 - self.beta2 ** state["step"])
        return m_hat / (torch.sqrt(v_hat).unsqueeze(1) + self.eps)

    def _update_block_v(self, state: dict, update: torch.Tensor, block: str) -> torch.Tensor:
        if block == "row" and update.ndim >= 2:
            key = "v_row"
            value = update.pow(2).reshape(update.size(0), -1).mean(dim=1)
            shape = (update.size(0),) + (1,) * (update.ndim - 1)
        elif block == "col" and update.ndim >= 2:
            key = "v_col"
            value = update.pow(2).transpose(0, 1).reshape(update.size(1), -1).mean(dim=1)
            shape = (1, update.size(1)) + (1,) * (update.ndim - 2)
        else:
            key = "v_full"
            value = update.pow(2).mean()
            shape = None

        if key not in state:
            state[key] = torch.zeros_like(value)
        state[key].mul_(self.beta2).add_(value, alpha=1.0 - self.beta2)

        if shape is None:
            return state[key]
        return state[key].view(shape)
