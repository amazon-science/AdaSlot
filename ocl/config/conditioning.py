"""Configuration of slot conditionings."""
import dataclasses

from hydra_zen import builds
from omegaconf import SI

from ocl import conditioning


@dataclasses.dataclass
class ConditioningConfig:
    """Base class for conditioning module configuration."""


# Unfortunately, we cannot define object_dim as part of the base config class as this prevents using
# required positional arguments in all subclasses. We thus instead pass them here.
LearntConditioningConfig = builds(
    conditioning.LearntConditioning,
    object_dim=SI("${perceptual_grouping.object_dim}"),
    builds_bases=(ConditioningConfig,),
    populate_full_signature=True,
)

RandomConditioningConfig = builds(
    conditioning.RandomConditioning,
    object_dim=SI("${perceptual_grouping.object_dim}"),
    builds_bases=(ConditioningConfig,),
    populate_full_signature=True,
)

RandomConditioningWithQMCSamplingConfig = builds(
    conditioning.RandomConditioningWithQMCSampling,
    object_dim=SI("${perceptual_grouping.object_dim}"),
    builds_bases=(ConditioningConfig,),
    populate_full_signature=True,
)

SlotwiseLearntConditioningConfig = builds(
    conditioning.SlotwiseLearntConditioning,
    object_dim=SI("${perceptual_grouping.object_dim}"),
    builds_bases=(ConditioningConfig,),
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="schemas", name="conditioning", node=ConditioningConfig)

    config_store.store(group="conditioning", name="learnt", node=LearntConditioningConfig)
    config_store.store(group="conditioning", name="random", node=RandomConditioningConfig)
    config_store.store(
        group="conditioning",
        name="random_with_qmc_sampling",
        node=RandomConditioningWithQMCSamplingConfig,
    )
    config_store.store(
        group="conditioning", name="slotwise_learnt_random", node=SlotwiseLearntConditioningConfig
    )
