"""Perceptual grouping models."""
import dataclasses

from hydra_zen import builds

from ocl import perceptual_grouping


@dataclasses.dataclass
class PerceptualGroupingConfig:
    """Configuration class of perceptual grouping models."""


SlotAttentionConfig = builds(
    perceptual_grouping.SlotAttentionGrouping,
    builds_bases=(PerceptualGroupingConfig,),
    populate_full_signature=True,
)
SlotAttentionGumbelV1Config = builds(
    perceptual_grouping.SlotAttentionGroupingGumbelV1,
    builds_bases=(PerceptualGroupingConfig,),
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="schemas", name="perceptual_grouping", node=PerceptualGroupingConfig)
    config_store.store(group="perceptual_grouping", name="slot_attention", node=SlotAttentionConfig)
    config_store.store(group="perceptual_grouping", name="slot_attention_gumbel_v1", node=SlotAttentionGumbelV1Config)
