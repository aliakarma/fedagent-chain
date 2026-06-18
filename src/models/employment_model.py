"""Core employment matching neural network for FedAgent-Chain.

Implements a multilayer perceptron for binary employment suitability classification.
The model is trained locally at each institutional node and its updates are
aggregated via federated averaging.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig


class EmploymentMatchingModel(nn.Module):
    """Feedforward neural network for employment suitability prediction.

    Architecture:
        Input (skill_vector + accommodation_needs + language_enc + edu_level)
        → Linear → BatchNorm → ReLU → Dropout
        → Linear → BatchNorm → ReLU → Dropout
        → Linear → Sigmoid (binary suitability probability)

    Input dimension: 50 (skills) + 20 (accommodation) + 8 (disability OHE)
                    + 4 (work mode OHE) + 5 (education) + 4 (employment goal) = 91

    Parameters
    ----------
    input_dim : int
        Feature dimension of the combined user-job input vector.
    hidden_dims : list of int
        Sizes of hidden layers.
    dropout_rate : float
        Dropout probability for regularization.
    """

    INPUT_DIM = 91  # Combined user + job feature dimension

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]  # paper: two hidden layers (256, 128)

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hdim),
                    nn.LayerNorm(hdim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout_rate),
                ]
            )
            prev_dim = hdim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Xavier uniform initialization to all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning suitability probability.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Suitability probability of shape (batch_size, 1), values in [0, 1].
        """
        return torch.sigmoid(self.network(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Predict binary suitability labels.

        Parameters
        ----------
        x : torch.Tensor
            Input feature tensor.
        threshold : float
            Decision threshold for binary classification.

        Returns
        -------
        torch.Tensor
            Binary predictions of shape (batch_size,).
        """
        with torch.no_grad():
            probs = self.forward(x).squeeze(-1)
            return (probs >= threshold).long()

    def get_state_dict_numpy(self) -> dict[str, np.ndarray]:
        """Return model state dict with numpy arrays (for federated communication).

        Returns
        -------
        dict
            Parameter name → numpy array mapping.
        """
        return {name: param.detach().cpu().numpy() for name, param in self.state_dict().items()}

    def load_state_dict_numpy(self, numpy_state: dict[str, np.ndarray]) -> None:
        """Load model weights from a numpy state dict.

        Parameters
        ----------
        numpy_state : dict
            Parameter name → numpy array mapping.
        """
        torch_state = {
            name: torch.from_numpy(np.array(arr).copy()) for name, arr in numpy_state.items()
        }
        self.load_state_dict(torch_state)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> EmploymentMatchingModel:
        """Instantiate a model from a Hydra config object.

        Parameters
        ----------
        cfg : DictConfig
            Configuration with fields: input_dim, hidden_dims, dropout_rate.

        Returns
        -------
        EmploymentMatchingModel
            Initialized model instance.
        """
        return cls(
            input_dim=cfg.get("input_dim", cls.INPUT_DIM),
            hidden_dims=list(cfg.get("hidden_dims", [256, 128])),
            dropout_rate=float(cfg.get("dropout_rate", 0.3)),
        )

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
