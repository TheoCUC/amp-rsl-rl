"""Placeholder utilities for loading ASE demonstration data."""

from typing import Generator, Tuple

import torch


class ASELoader:
    """Minimal data loader for ASE expert demonstrations.

    The loader yields dummy state transitions and is intended to be
    replaced with a proper dataset handling implementation.
    """

    def __init__(self, obs_dim: int, device: str = "cpu") -> None:
        self.obs_dim = obs_dim
        self.device = torch.device(device)

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """Yields batches of random (state, next_state) pairs."""
        for _ in range(num_mini_batch):
            state = torch.randn(mini_batch_size, self.obs_dim, device=self.device)
            next_state = torch.randn(mini_batch_size, self.obs_dim, device=self.device)
            yield state, next_state
