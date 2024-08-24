from typing import Any, Callable, Dict, Tuple

import webdataset
from pluggy import HookimplMarker, HookspecMarker

from ocl.combined_model import CombinedModel

hook_specification = HookspecMarker("ocl")
hook_implementation = HookimplMarker("ocl")


class FakeHooks:
    """Class that mimics the behavior of the plugin manager hooks property."""

    def __getattr__(self, attribute):
        """Return a fake hook handler for any attribute query."""

        def fake_hook_handler(*args, **kwargs):
            return tuple()

        return fake_hook_handler


# @transform_hooks
# def input_dependencies() -> Tuple[str, ...]:
#     """Provide list of variables that are required for the plugin to function."""
#
#
# @transform_hooks
# def provided_inputs() -> Tuple[str, ...]:
#     """Provide list of variables that are provided by the plugin."""


@hook_specification
def training_transform() -> Callable[[webdataset.Processor], webdataset.Processor]:
    """Provide a transformation which processes a component of a webdataset pipeline."""


@hook_specification
def training_batch_transform() -> Callable[[webdataset.Processor], webdataset.Processor]:
    """Provide a transformation which processes a batched component of a webdataset pipeline."""


@hook_specification
def training_fields() -> Tuple[str]:
    """Provide list of fields that are required to be decoded during training."""


@hook_specification
def evaluation_transform() -> Callable[[webdataset.Processor], webdataset.Processor]:
    """Provide a transformation which processes a component of a webdataset pipeline."""


@hook_specification
def evaluation_batch_transform() -> Callable[[webdataset.Processor], webdataset.Processor]:
    """Provide a transformation which processes a batched component of a webdataset pipeline."""


@hook_specification
def evaluation_fields() -> Tuple[str]:
    """Provide list of fields that are required to be decoded during evaluation."""


@hook_specification
def configure_optimizers(model: CombinedModel) -> Dict[str, Any]:
    """Return optimizers in the format of pytorch lightning."""


@hook_specification
def on_train_start(model: CombinedModel) -> None:
    """Hook called when starting training."""


@hook_specification
def on_train_epoch_start(model: CombinedModel) -> None:
    """Hook called when starting training epoch."""
