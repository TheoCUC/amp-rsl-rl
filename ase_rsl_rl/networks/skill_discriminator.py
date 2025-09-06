"""Skill-aware discriminator for ASE."""

from amp_rsl_rl.networks import Discriminator


class SkillDiscriminator(Discriminator):
    """Discriminator tailored for Adversarial Skill Embeddings.

    This class currently inherits all functionality from
    :class:`amp_rsl_rl.networks.Discriminator` but is provided as a
    dedicated entry point for ASE-specific extensions.
    """

    pass
