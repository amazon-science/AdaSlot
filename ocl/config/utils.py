"""Utility functions useful for configuration."""
import ast
from typing import Any, Callable

from hydra_zen import builds

from ocl.config.feature_extractors import FeatureExtractorConfig
from ocl.config.perceptual_groupings import PerceptualGroupingConfig
from ocl.distillation import EMASelfDistillation
from ocl.utils.masking import CreateSlotMask
from ocl.utils.routing import Combined, Recurrent
import torch

def lambda_string_to_function(function_string: str) -> Callable[..., Any]:
    """Convert string of the form "lambda x: x" into a callable Python function."""
    # This is a bit hacky but ensures that the syntax of the input is correct and contains
    # a valid lambda function definition without requiring to run `eval`.
    parsed = ast.parse(function_string)
    is_lambda = isinstance(parsed.body[0], ast.Expr) and isinstance(parsed.body[0].value, ast.Lambda)
    if not is_lambda:
        raise ValueError(f"'{function_string}' is not a valid lambda definition.")

    return eval(function_string)


class ConfigDefinedLambda:
    """Lambda function defined in the config.

    This allows lambda functions defined in the config to be pickled.
    """

    def __init__(self, function_string: str):
        self.__setstate__(function_string)

    def __getstate__(self) -> str:
        return self.function_string

    def __setstate__(self, function_string: str):
        self.function_string = function_string
        self._fn = lambda_string_to_function(function_string)

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def eval_lambda(function_string, *args):
    lambda_fn = lambda_string_to_function(function_string)
    return lambda_fn(*args)


FunctionConfig = builds(ConfigDefinedLambda, populate_full_signature=True)

# Inherit from all so it can be used in place of any module.
CombinedConfig = builds(
    Combined,
    populate_full_signature=True,
    builds_bases=(FeatureExtractorConfig, PerceptualGroupingConfig),
)
RecurrentConfig = builds(
    Recurrent,
    populate_full_signature=True,
    builds_bases=(FeatureExtractorConfig, PerceptualGroupingConfig),
)
CreateSlotMaskConfig = builds(CreateSlotMask, populate_full_signature=True)


EMASelfDistillationConfig = builds(
    EMASelfDistillation,
    populate_full_signature=True,
    builds_bases=(FeatureExtractorConfig, PerceptualGroupingConfig),
)


def make_slice(expr):
    if isinstance(expr, int):
        return expr

    pieces = [s and int(s) or None for s in expr.split(":")]
    if len(pieces) == 1:
        return slice(pieces[0], pieces[0] + 1)
    else:
        return slice(*pieces)


def slice_string(string: str, split_char: str, slice_str: str) -> str:
    """Split a string according to a split_char and slice.

    If the output contains more than one element, join these using the split char again.
    """
    sl = make_slice(slice_str)
    res = string.split(split_char)[sl]
    if isinstance(res, list):
        res = split_char.join(res)
    return res


def register_configs(config_store):
    config_store.store(group="schemas", name="lambda_fn", node=FunctionConfig)
    config_store.store(group="utils", name="combined", node=CombinedConfig)
    config_store.store(group="utils", name="selfdistillation", node=EMASelfDistillationConfig)
    config_store.store(group="utils", name="recurrent", node=RecurrentConfig)
    config_store.store(group="utils", name="create_slot_mask", node=CreateSlotMaskConfig)


def register_resolvers(omegaconf):
    omegaconf.register_new_resolver("lambda_fn", ConfigDefinedLambda)
    omegaconf.register_new_resolver("eval_lambda", eval_lambda)
    omegaconf.register_new_resolver("slice", slice_string)
