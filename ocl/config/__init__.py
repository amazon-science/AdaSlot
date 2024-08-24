from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from ocl.config import (
    conditioning,
    datasets,
    feature_extractors,
    metrics,
    neural_networks,
    optimizers,
    perceptual_groupings,
    plugins,
    predictor,
    utils,
)

config_store = ConfigStore.instance()

conditioning.register_configs(config_store)

datasets.register_configs(config_store)
datasets.register_resolvers(OmegaConf)

feature_extractors.register_configs(config_store)

metrics.register_configs(config_store)

neural_networks.register_configs(config_store)

optimizers.register_configs(config_store)

perceptual_groupings.register_configs(config_store)
predictor.register_configs(config_store)

plugins.register_configs(config_store)
plugins.register_resolvers(OmegaConf)

utils.register_configs(config_store)
utils.register_resolvers(OmegaConf)
