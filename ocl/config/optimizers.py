"""Pytorch optimizers."""
import dataclasses

import torch.optim
from hydra_zen import make_custom_builds_fn


@dataclasses.dataclass
class OptimizerConfig:
    pass


# TODO(hornmax): We cannot automatically extract type information from the torch SGD implementation,
# thus we define it manually here.
@dataclasses.dataclass
class SGDConfig(OptimizerConfig):
    learning_rate: float
    momentum: float = 0.0
    dampening: float = 0.0
    nestov: bool = False
    _target_: str = "hydra_zen.funcs.zen_processing"
    _zen_target: str = "torch.optim.SGD"
    _zen_partial: bool = True


pbuilds = make_custom_builds_fn(
    zen_partial=True,
    populate_full_signature=True,
)

AdamConfig = pbuilds(torch.optim.Adam, builds_bases=(OptimizerConfig,))
AdamWConfig = pbuilds(torch.optim.AdamW, builds_bases=(OptimizerConfig,))


def register_configs(config_store):
    config_store.store(group="optimizers", name="sgd", node=SGDConfig)
    config_store.store(group="optimizers", name="adam", node=AdamConfig)
    config_store.store(group="optimizers", name="adamw", node=AdamWConfig)
