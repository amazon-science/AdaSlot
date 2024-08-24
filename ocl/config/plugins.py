"""Configuration of plugins."""
import dataclasses
import functools

import hydra_zen
from hydra_zen import builds
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from ocl import plugins, scheduling
from ocl.config.optimizers import OptimizerConfig


@dataclasses.dataclass
class PluginConfig:
    """Base class for plugin configurations."""

    pass


@dataclasses.dataclass
class LRSchedulerConfig:
    pass


def exponential_decay_with_optional_warmup(
    optimizer, decay_rate: float = 1.0, decay_steps: int = 10000, warmup_steps: int = 0
):
    """Return pytorch lighting optimizer configuration for exponential decay with optional warmup.

    Returns:
        Dict with structure compatible with ptl.  See
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    """
    decay_fn = functools.partial(
        scheduling.exp_decay_with_warmup_fn,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
    )

    return {"lr_scheduler": {"scheduler": LambdaLR(optimizer, decay_fn), "interval": "step"}}


def exponential_decay_after_optional_warmup(
    optimizer, decay_rate: float = 1.0, decay_steps: int = 10000, warmup_steps: int = 0
):
    """Return pytorch lighting optimizer configuration for exponential decay after optional warmup.

    Returns:
        Dict with structure compatible with ptl.  See
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    """
    decay_fn = functools.partial(
        scheduling.exp_decay_after_warmup_fn,
        decay_rate=decay_rate,
        decay_steps=decay_steps,
        warmup_steps=warmup_steps,
    )

    return {"lr_scheduler": {"scheduler": LambdaLR(optimizer, decay_fn), "interval": "step"}}


def plateau_decay(optimizer, decay_rate: float = 1.0, patience: int = 10):
    """Return pytorch lighting optimizer configuration for plato decay.

    Returns:
        Dict with structure compatible with ptl.  See
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    """
    plateau_scheduler = ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=decay_rate, patience=patience
    )
    return {
        "lr_scheduler": {
            "scheduler": plateau_scheduler,
            "interval": "epoch",
            "monitor": "val/loss_total",
        }
    }


def cosine_annealing_with_optional_warmup(
    optimizer,
    T_max: int = 100000,
    eta_min: float = 0.0,
    warmup_steps: int = 0,
    error_on_exceeding_steps: bool = True,
):
    """Return pytorch lighting optimizer configuration for cosine annealing with warmup.

    Returns:
        Dict with structure compatible with ptl.  See
        https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
    """
    return {
        "lr_scheduler": {
            "scheduler": scheduling.CosineAnnealingWithWarmup(
                optimizer,
                T_max,
                eta_min=eta_min,
                warmup_steps=warmup_steps,
                error_on_exceeding_steps=error_on_exceeding_steps,
            ),
            "interval": "step",
        }
    }


PlateauDecayLR = builds(
    plateau_decay,
    zen_partial=True,
    populate_full_signature=True,
    builds_bases=(LRSchedulerConfig,),
)

ExpDecayLR = builds(
    exponential_decay_with_optional_warmup,
    zen_partial=True,
    populate_full_signature=True,
    builds_bases=(LRSchedulerConfig,),
)

ExpDecayAfterWarmupLR = builds(
    exponential_decay_after_optional_warmup,
    zen_partial=True,
    populate_full_signature=True,
    builds_bases=(LRSchedulerConfig,),
)

CosineAnnealingLR = builds(
    cosine_annealing_with_optional_warmup,
    zen_partial=True,
    populate_full_signature=True,
    builds_bases=(LRSchedulerConfig,),
)


@dataclasses.dataclass
class HPSchedulerConfig:
    """Base class for hyperparameter scheduler configuration."""


LinearHPSchedulerConfig = builds(
    scheduling.LinearHPScheduler,
    builds_bases=(HPSchedulerConfig,),
    populate_full_signature=True,
)
StepHPSchedulerConfig = builds(
    scheduling.StepHPScheduler,
    builds_bases=(HPSchedulerConfig,),
    populate_full_signature=True,
)
CosineAnnealingHPSchedulerConfig = builds(
    scheduling.CosineAnnealingHPScheduler,
    builds_bases=(HPSchedulerConfig,),
    populate_full_signature=True,
)


builds_plugin = hydra_zen.make_custom_builds_fn(
    populate_full_signature=True,
)
OptimizationConfig = builds_plugin(
    plugins.Optimization,
    optimizer=OptimizerConfig,
    lr_scheduler=LRSchedulerConfig,
    builds_bases=(PluginConfig,),
)
SingleElementPreprocessingConfig = builds_plugin(
    plugins.SingleElementPreprocessing, builds_bases=(PluginConfig,)
)
MultiElementPreprocessingConfig = builds_plugin(
    plugins.MultiElementPreprocessing, builds_bases=(PluginConfig,)
)
DataPreprocessingConfig = builds_plugin(plugins.DataPreprocessing, builds_bases=(PluginConfig,))
SubsetDatasetConfig = builds_plugin(plugins.SubsetDataset, builds_bases=(PluginConfig,))
SampleFramesFromVideoConfig = builds_plugin(
    plugins.SampleFramesFromVideo, builds_bases=(PluginConfig,)
)
SplitConsecutiveFramesConfig = builds_plugin(
    plugins.SplitConsecutiveFrames, builds_bases=(PluginConfig,)
)
RandomStridedWindowConfig = builds_plugin(plugins.RandomStridedWindow, builds_bases=(PluginConfig,))
RenameFieldsConfig = builds_plugin(plugins.RenameFields, builds_bases=(PluginConfig,))
SpatialSlidingWindowConfig = builds_plugin(
    plugins.SpatialSlidingWindow, builds_bases=(PluginConfig,)
)


def register_configs(config_store):
    config_store.store(group="schemas", name="lr_scheduler", node=LRSchedulerConfig)
    config_store.store(group="lr_schedulers", name="exponential_decay", node=ExpDecayLR)
    config_store.store(
        group="lr_schedulers", name="exponential_decay_after_warmup", node=ExpDecayAfterWarmupLR
    )
    config_store.store(group="lr_schedulers", name="plateau_decay", node=PlateauDecayLR)
    config_store.store(group="lr_schedulers", name="cosine_annealing", node=CosineAnnealingLR)

    config_store.store(group="schemas", name="hp_scheduler", node=HPSchedulerConfig)
    config_store.store(group="hp_schedulers", name="linear", node=LinearHPSchedulerConfig)
    config_store.store(group="hp_schedulers", name="step", node=StepHPSchedulerConfig)
    config_store.store(
        group="hp_schedulers", name="cosine_annealing", node=CosineAnnealingHPSchedulerConfig
    )

    config_store.store(group="schemas", name="plugin", node=PluginConfig)
    config_store.store(group="plugins", name="optimization", node=OptimizationConfig)
    config_store.store(
        group="plugins",
        name="single_element_preprocessing",
        node=SingleElementPreprocessingConfig,
    )
    config_store.store(
        group="plugins",
        name="multi_element_preprocessing",
        node=MultiElementPreprocessingConfig,
    )
    config_store.store(
        group="plugins",
        name="data_preprocessing",
        node=DataPreprocessingConfig,
    )
    config_store.store(group="plugins", name="subset_dataset", node=SubsetDatasetConfig)
    config_store.store(
        group="plugins", name="sample_frames_from_video", node=SampleFramesFromVideoConfig
    )
    config_store.store(
        group="plugins", name="split_consecutive_frames", node=SplitConsecutiveFramesConfig
    )
    config_store.store(group="plugins", name="random_strided_window", node=RandomStridedWindowConfig)
    config_store.store(group="plugins", name="rename_fields", node=RenameFieldsConfig)
    config_store.store(
        group="plugins", name="spatial_sliding_window", node=SpatialSlidingWindowConfig
    )


def _torchvision_interpolation_mode(mode):
    import torchvision

    return torchvision.transforms.InterpolationMode[mode.upper()]


def register_resolvers(omegaconf):
    omegaconf.register_new_resolver(
        "torchvision_interpolation_mode", _torchvision_interpolation_mode
    )
