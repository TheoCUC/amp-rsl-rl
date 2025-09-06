"""ASE integrated with PPO for the RSL-RL framework."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from amp_rsl_rl.algorithms.amp_ppo import AMP_PPO
from ase_rsl_rl.networks import SkillDiscriminator
from ase_rsl_rl.storage import SkillReplayBuffer
from ase_rsl_rl.utils import ASELoader


class ASE_PPO(AMP_PPO):
    """PPO algorithm augmented with Adversarial Skill Embeddings.

    This class inherits from :class:`amp_rsl_rl.algorithms.AMP_PPO` and
    exposes an interface for skill-conditioned policies.  The current
    implementation mirrors the behaviour of ``AMP_PPO`` and acts as a
    scaffold for future ASE-specific logic.
    """

    def __init__(
        self,
        actor_critic,
        discriminator: SkillDiscriminator,
        ase_data: ASELoader,
        ase_normalizer: Optional[Any],
        skill_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            actor_critic=actor_critic,
            discriminator=discriminator,
            amp_data=ase_data,
            amp_normalizer=ase_normalizer,
            **kwargs,
        )
        self.skill_dim = skill_dim

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Tuple[int, ...],
        critic_obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        skill_shape: Tuple[int, ...],
    ) -> None:
        """Initialises rollout storage and reserves space for skill latents."""
        super().init_storage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            actor_obs_shape=actor_obs_shape,
            critic_obs_shape=critic_obs_shape,
            action_shape=action_shape,
        )
        assert self.storage is not None
        self.storage.extras = {
            "skills": torch.zeros(
                (num_transitions_per_env, num_envs, *skill_shape), device=self.device
            )
        }

    def act(
        self, obs: torch.Tensor, critic_obs: torch.Tensor, skill: torch.Tensor
    ) -> torch.Tensor:
        """Selects actions conditioned on the provided skill embedding.

        The default behaviour simply forwards to ``AMP_PPO.act``.
        """
        return super().act(obs, critic_obs)
