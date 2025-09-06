"""Replay buffer storing transitions augmented with skill embeddings."""

from amp_rsl_rl.storage import ReplayBuffer


class SkillReplayBuffer(ReplayBuffer):
    """Replay buffer used by ASE to cache policy transitions.

    The implementation currently mirrors
    :class:`amp_rsl_rl.storage.ReplayBuffer` and acts as a placeholder
    for potential ASE-specific extensions.
    """

    pass
