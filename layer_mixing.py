"""Layer projection and mixing modules for speech attribute representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerLinearProjectMix(nn.Module):
    """
    Project each selected encoder layer, then learn a weighted combination.

    Each layer has a full linear projection. Softmax-normalized logits weight
    the projected features. An optional layer normalization follows the mixture.

    Args:
        num_layers (int): Expected number of input hidden states.
        d_model (int): Input and output feature dimension.
        layer_start (int): Index of the first selected layer.
        layer_end (int, optional): Index after the last selected layer
        use_layernorm (bool): Apply layer normalization to the mixture.
        init_identity (bool): Initialize every projection to the identity.

    Shape:
        Input: ``[num_layers, batch, time, d_model]``.
        Output: ``[batch, time, d_model]``.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        layer_start: int = 0,
        layer_end: int = None,
        use_layernorm: bool = True,
        init_identity: bool = True,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        self.layer_start = int(layer_start)
        self.layer_end = int(layer_end) if layer_end is not None else int(num_layers)

        self.d_model = int(d_model)
        if self.num_layers <= 0 or self.d_model <= 0:
            raise ValueError("num_layers and d_model must be positive.")
        if not 0 <= self.layer_start < self.layer_end <= self.num_layers:
            raise ValueError(
                "Layer bounds must satisfy "
                "0 <= layer_start < layer_end <= num_layers."
            )

        n = self.layer_end - self.layer_start
        self.n = n

        # Uniform initial mixing weights over the selected layers.
        self.logits = nn.Parameter(torch.zeros(n))  # softmax weights

        # Per-layer full projection: W[l] is [D,D], b[l] is [D]
        self.W = nn.Parameter(torch.empty(n, d_model, d_model))
        self.b = nn.Parameter(torch.zeros(n, d_model))

        # Identity initialization preserves features before mixing.
        if init_identity:
            with torch.no_grad():
                self.W.zero_()
                idx = torch.arange(d_model)
                self.W[:, idx, idx] = 1.0
                self.b.zero_()
        else:
            nn.init.xavier_uniform_(self.W)
            nn.init.zeros_(self.b)

        self.ln = nn.LayerNorm(d_model) if use_layernorm else None

    def forward(self, feats_all: torch.Tensor) -> torch.Tensor:
        """Combine the selected hidden states into frame-level features."""
        if not torch.is_tensor(feats_all) or feats_all.ndim != 4:
            raise ValueError("Expected hidden states with shape [L, B, T, D].")
        if feats_all.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected feature dimension {self.d_model}, "
                f"got {feats_all.shape[-1]}."
            )
        # feats_all: [L,B,T,D]
        L = feats_all.shape[0]
        if L != self.num_layers:
            raise RuntimeError(
                f"LayerLinearProjectMix expected num_layers={self.num_layers}, got L={L}. "
                f"Fix num_w2v_layers in YAML to match feats_all.shape[0]."
            )

        feats = feats_all[self.layer_start:self.layer_end]  
        w = F.softmax(self.logits, dim=0)                 

        # proj[l,b,t,h] = sum_d feats[l,b,t,d] * W[l,h,d] + b[l,h]
        proj = (
            torch.einsum("lbtd,lhd->lbth", feats, self.W)
            + self.b[:, None, None, :]
        )  # [n, B, T, D]

        # mix -> [B,T,D]
        mixed = torch.einsum("l,lbtd->btd", w, proj)

        if self.ln is not None:
            mixed = self.ln(mixed)
        return mixed
