"""Perceptual grouping models."""
import dataclasses

from hydra_zen import builds

from ocl import predictor


@dataclasses.dataclass
class PredictorConfig:
    """Configuration class of Predictor."""


TransitionConfig = builds(
    predictor.Predictor,
    builds_bases=(PredictorConfig,),
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="schemas", name="predictor", node=PredictorConfig)
    config_store.store(group="predictor", name="multihead_attention", node=TransitionConfig)
