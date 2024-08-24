import functools
import logging
import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import decord
import numpy as np
import torch
import webdataset
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

from ocl import hooks
from ocl.utils.trees import get_tree_element

decord.bridge.set_bridge("torch")
LOGGER = logging.getLogger(__name__)


class Plugin:
    """A plugin which defines a set of hooks to be called by the code."""


class Optimization(Plugin):
    """Optimize (a subset of) the parameters using a optimizer and a LR scheduler."""

    def __init__(
        self, optimizer, lr_scheduler=None, parameter_groups: Optional[List[Dict[str, Any]]] = None
    ):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.parameter_group_specs = parameter_groups
        if self.parameter_group_specs:
            for idx, param_group_spec in enumerate(self.parameter_group_specs):
                if "params" not in param_group_spec:
                    raise ValueError(f'Parameter group {idx + 1} does not contain key "params"')
                param_spec = param_group_spec["params"]
                if isinstance(param_spec, str):
                    param_group_spec["params"] = [param_spec]
                elif isinstance(param_spec, Iterable):
                    param_group_spec["params"] = list(param_spec)
                else:
                    raise ValueError(
                        f'"params" for parameter group {idx + 1} is not of type str or iterable'
                    )

                if "predicate" in param_group_spec:
                    if not callable(param_group_spec["predicate"]):
                        raise ValueError(
                            f'"predicate" for parameter group {idx + 1} is not a callable'
                        )

    def _get_parameter_groups(self, model):
        """Build parameter groups from specification."""
        parameter_groups = []
        for param_group_spec in self.parameter_group_specs:
            param_spec = param_group_spec["params"]
            # Default predicate includes all parameters
            predicate = param_group_spec.get("predicate", lambda name, param: True)

            parameters = []
            for parameter_path in param_spec:
                root = model
                for child in parameter_path.split("."):
                    root = getattr(root, child)
                parameters.extend(
                    param for name, param in root.named_parameters() if predicate(name, param)
                )

            param_group = {
                k: v for k, v in param_group_spec.items() if k not in ("params", "predicate")
            }
            param_group["params"] = parameters
            parameter_groups.append(param_group)

        return parameter_groups

    @hooks.hook_implementation
    def configure_optimizers(self, model):
        if self.parameter_group_specs:
            params_or_param_groups = self._get_parameter_groups(model)
        else:
            params_or_param_groups = model.parameters()

        optimizer = self.optimizer(params_or_param_groups)
        output = {"optimizer": optimizer}
        if self.lr_scheduler:
            output.update(self.lr_scheduler(optimizer))
        return output


class FreezeParameters(Plugin):
    def __init__(self, parameter_groups: List[Dict[str, Any]]):
        self.parameter_group_specs = parameter_groups
        for idx, param_group_spec in enumerate(self.parameter_group_specs):
            if "params" not in param_group_spec:
                raise ValueError(f'Parameter group {idx + 1} does not contain key "params"')
            param_spec = param_group_spec["params"]
            if isinstance(param_spec, str):
                param_group_spec["params"] = [param_spec]
            elif isinstance(param_spec, Iterable):
                param_group_spec["params"] = list(param_spec)
            else:
                raise ValueError(
                    f'"params" for parameter group {idx + 1} is not of type str or iterable'
                )

            if "predicate" in param_group_spec:
                if not callable(param_group_spec["predicate"]):
                    raise ValueError(f'"predicate" for parameter group {idx + 1} is not a callable')

    def _get_parameters_to_freeze(self, model):
        """Build parameter groups from specification."""
        parameters_to_freeze = []
        for param_group_spec in self.parameter_group_specs:
            for current_params in param_group_spec["params"]:
                param_path = current_params.split(".")
                # Default predicate includes all parameters
                predicate = param_group_spec.get("predicate", lambda name, param: True)
                param = get_tree_element(model, param_path)
                if isinstance(param, torch.nn.Module):
                    parameters_to_freeze.extend(
                        param for name, param in param.named_parameters() if predicate(name, param)
                    )
                elif isinstance(param, torch.nn.Parameter):
                    parameters_to_freeze.append(param)
                else:
                    raise ValueError(
                        "Object at path {'.'.join(param_path)} is neither nn.Module nor nn.Parameter"
                    )
        return parameters_to_freeze

    @hooks.hook_implementation
    def on_train_start(self, model):
        parameters_to_freeze = self._get_parameters_to_freeze(model)
        for param in parameters_to_freeze:
            param.requires_grad_(False)


class RestoreParameterSubset(Plugin):
    """Restore a subset of parameters using a checkpoint form a different model."""

    def __init__(self, checkpoint_file: str, target_path: str, source_path: Optional[str] = None):
        self.checkpoint_file = checkpoint_file
        self.target_path = target_path
        self.source_path = source_path if source_path else self.target_path

    @hooks.hook_implementation
    def on_train_start(self, model):
        if model.global_step != 0:
            # Don't restore when we are resuming training.
            rank_zero_warn("Not restoring parameter subset as training is being resumed")
            return
        device = model.device
        # Get parameters from state dict, load to cpu first to avoid memory issues.
        state_dict = torch.load(self.checkpoint_file, map_location="cpu")["state_dict"]
        # Add offset of 1 to remove potential dot.
        offset_keys = len(self.source_path) + 1
        state_dict = {
            key[offset_keys:]: value
            for key, value in state_dict.items()
            if key.startswith(self.source_path)
        }

        # Get module from model
        model_component: torch.nn.Module = get_tree_element(model, self.target_path.split("."))
        result = model_component.load_state_dict(state_dict, strict=False)
        model_component.to(device=device)
        if len(result.missing_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Missing keys: {result.missing_keys}"
            )
        if len(result.unexpected_keys):
            rank_zero_warn(
                f"Mismatch between state dict and model. Unexpected keys: {result.missing_keys}"
            )


def transform_with_duplicate(elements: dict, *, transform, element_key: str, duplicate_key: str):
    """Utility function to fix issues with pickling."""
    element = transform(elements[element_key])
    elements[element_key] = element
    elements[duplicate_key] = element
    return elements


class SingleElementPreprocessing(Plugin):
    """Preprocessing of a single element in the input data.

    This is useful to build preprocessing pipelines based on existing element transformations such
    as those provided by torchvision. The element can optionally be duplicated and stored under a
    different key after the transformation by specifying `duplicate_key`. This is useful to further
    preprocess this element in different ways afterwards.
    """

    def __init__(
        self,
        training_transform: Callable,
        evaluation_transform: Callable,
        element_key: str = "image",
        duplicate_key: Optional[str] = None,
    ):
        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self.element_key = element_key
        self.duplicate_key = duplicate_key

    @hooks.hook_implementation
    def training_fields(self):
        return (self.element_key,)

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transform:

            if self.duplicate_key is None:

                def transform(pipeline: webdataset.Processor):
                    return pipeline.map_dict(**{self.element_key: self._training_transform})

            else:

                def transform(pipeline: webdataset.Processor):
                    transform_func = functools.partial(
                        transform_with_duplicate,
                        transform=self._training_transform,
                        element_key=self.element_key,
                        duplicate_key=self.duplicate_key,
                    )
                    return pipeline.map(transform_func)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return (self.element_key,)

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transform:

            if self.duplicate_key is None:

                def transform(pipeline: webdataset.Processor):
                    return pipeline.map_dict(**{self.element_key: self._evaluation_transform})

            else:

                def transform(pipeline: webdataset.Processor):
                    transform_func = functools.partial(
                        transform_with_duplicate,
                        transform=self._evaluation_transform,
                        element_key=self.element_key,
                        duplicate_key=self.duplicate_key,
                    )

                    return pipeline.map(transform_func)

            return transform
        else:
            return None


class MultiElementPreprocessing(Plugin):
    """Preprocessing of multiple elements in the input data.

    This is useful preprocessing pipelines based on existing element transformations such as those
    provided by torchvision.
    """

    def __init__(
        self,
        training_transforms: Optional[Dict[str, Any]] = None,
        evaluation_transforms: Optional[Dict[str, Any]] = None,
    ):
        if training_transforms is None:
            training_transforms = {}
        self.training_keys = tuple(training_transforms)
        self._training_transforms = {
            key: transf for key, transf in training_transforms.items() if transf is not None
        }

        if evaluation_transforms is None:
            evaluation_transforms = {}
        self.evaluation_keys = tuple(evaluation_transforms)
        self._evaluation_transforms = {
            key: transf for key, transf in evaluation_transforms.items() if transf is not None
        }

    @hooks.hook_implementation
    def training_fields(self):
        return self.training_keys

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transforms:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map_dict(**self._training_transforms)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self.evaluation_keys

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transforms:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map_dict(**self._evaluation_transforms)

            return transform
        else:
            return None


class DataPreprocessing(Plugin):
    """Arbitrary preprocessing of input data.

    The transform takes in a dictionary of elements and should return a dictionary of elements.
    Plugin must specify the elements that should be included in the dictionary using
    `training_fields` and `evaluation_fields` arguments.
    """

    def __init__(
        self,
        training_transform: Optional[Callable] = None,
        evaluation_transform: Optional[Callable] = None,
        training_fields: Optional[Sequence[str]] = None,
        evaluation_fields: Optional[Sequence[str]] = None,
    ):
        if training_transform is not None and training_fields is None:
            raise ValueError(
                "If passing `training_transform`, `training_fields` must also be specified."
            )
        if evaluation_transform is not None and evaluation_fields is None:
            raise ValueError(
                "If passing `evaluation_transform`, `evaluation_fields` must also be specified."
            )

        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self._training_fields = tuple(training_fields) if training_fields else tuple()
        self._evaluation_fields = tuple(evaluation_fields) if evaluation_fields else tuple()

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._training_transform)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self._evaluation_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._evaluation_transform)

            return transform
        else:
            return None


class BatchDataPreprocessing(Plugin):
    """Arbitrary preprocessing of input data batch.

    The transform takes in a dictionary of elements and should return a dictionary of elements.
    Plugin must specify the elements that should be included in the dictionary using
    `training_fields` and `evaluation_fields` arguments.
    """

    def __init__(
        self,
        training_transform: Optional[Callable] = None,
        evaluation_transform: Optional[Callable] = None,
        training_fields: Optional[Sequence[str]] = None,
        evaluation_fields: Optional[Sequence[str]] = None,
    ):
        if training_transform is not None and training_fields is None:
            raise ValueError(
                "If passing `training_transform`, `training_fields` must also be specified."
            )
        if evaluation_transform is not None and evaluation_fields is None:
            raise ValueError(
                "If passing `evaluation_transform`, `evaluation_fields` must also be specified."
            )

        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self._training_fields = tuple(training_fields) if training_fields else tuple()
        self._evaluation_fields = tuple(evaluation_fields) if evaluation_fields else tuple()

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_batch_transform(self):
        if self._training_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._training_transform)

            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_batch_transform(self):
        if self._evaluation_transform:

            def transform(pipeline: webdataset.Processor):
                return pipeline.map(self._evaluation_transform)

            return transform
        else:
            return None


def _transform_elements(inputs, transforms):
    for key, transform in transforms.items():
        inputs[key] = transform(inputs[key])
    return inputs


class MultiElementBatchPreprocessing(BatchDataPreprocessing):
    """Preprocessing of multiple elements in the batched input data.

    This is useful preprocessing pipelines based on existing element transformations such as those
    provided by torchvision.
    """

    def __init__(
        self,
        training_transforms: Optional[Dict[str, Any]] = None,
        evaluation_transforms: Optional[Dict[str, Any]] = None,
    ):
        if training_transforms is None:
            training_transform = None
            training_fields = None
        else:
            training_fields = tuple(training_transforms)
            training_transforms = {
                key: transf for key, transf in training_transforms.items() if transf is not None
            }
            training_transform = functools.partial(
                _transform_elements, transforms=training_transforms
            )

        if evaluation_transforms is None:
            evaluation_transform = None
            evaluation_fields = None
        else:
            evaluation_fields = tuple(evaluation_transforms)
            evaluation_transforms = {
                key: transf for key, transf in evaluation_transforms.items() if transf is not None
            }
            evaluation_transform = functools.partial(
                _transform_elements, transforms=evaluation_transforms
            )
        super().__init__(
            training_transform=training_transform,
            evaluation_transform=evaluation_transform,
            training_fields=training_fields,
            evaluation_fields=evaluation_fields,
        )


def _transform_single_element(inputs, field, transform, duplicate_key):
    # print("in _transform_single_element")
    if duplicate_key:
        inputs[duplicate_key] = inputs[field]
    inputs[field] = transform(inputs[field])
    return inputs


class SingleElementBatchPreprocessing(BatchDataPreprocessing):
    """Preprocessing of a single element in the batched input data.

    This is useful to build preprocessing pipelines based on existing element transformations such
    as those provided by torchvision. The element can optionally be duplicated and stored under a
    different key after the transformation by specifying `duplicate_key`. This is useful to further
    preprocess this element in different ways afterwards.
    """

    def __init__(
        self,
        training_transform: Optional[Callable],
        evaluation_transform: Optional[Callable],
        element_key: str = "image",
        duplicate_key: Optional[str] = None,
    ):
        if training_transform is None:
            training_fields = None
        else:
            training_fields = [element_key]
            training_transform = functools.partial(
                _transform_single_element,
                field=element_key,
                transform=training_transform,
                duplicate_key=duplicate_key,
            )

        if evaluation_transform is None:
            evaluation_fields = None
        else:
            evaluation_fields = [element_key]
            evaluation_transform = functools.partial(
                _transform_single_element,
                field=element_key,
                transform=evaluation_transform,
                duplicate_key=duplicate_key,
            )

        super().__init__(
            training_transform=training_transform,
            evaluation_transform=evaluation_transform,
            training_fields=training_fields,
            evaluation_fields=evaluation_fields,
        )


class SubsetDataset(Plugin):
    """Create a subset of a dataset by discarding samples."""

    def __init__(
        self, predicate, fields: Sequence[str], subset_train: bool = True, subset_eval: bool = True
    ):
        """Plugin to create a subset of a dataset by discarding samples.

        Args:
            predicate: Function which determines if elements should be kept (return value is True)
                or discarded (return value is False). The function is only provided with the fields
                specified in the `fields` parameter.
            fields (Sequence[str]): The fields from the input which should be passed on to the
                predicate for evaluation.
            subset_train: Subset training data.
            subset_eval: Subset evaluation data.
        """
        self.predicate = predicate
        self.fields = tuple(fields)
        self.subset_train = subset_train
        self.subset_eval = subset_eval

    def _get_transform_function(self):
        def wrapped_predicate(d: dict):
            return self.predicate(*(d[field] for field in self.fields))

        def select(pipeline: webdataset.Processor):
            return pipeline.select(wrapped_predicate)

        return select

    @hooks.hook_implementation
    def training_fields(self):
        if self.subset_train:
            return self.fields
        else:
            return tuple()

    @hooks.hook_implementation
    def training_transform(self):
        if self.subset_train:
            return self._get_transform_function()
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        if self.subset_eval:
            return self.fields
        else:
            return tuple()

    @hooks.hook_implementation
    def evaluation_transform(self):
        if self.subset_eval:
            return self._get_transform_function()
        else:
            return None


class SampleFramesFromVideo(Plugin):
    def __init__(
        self,
        n_frames_per_video: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
        seed: int = 39480234,
        per_epoch: bool = False,
        shuffle_buffer_size: int = 1000,
        n_eval_frames_per_video: Optional[int] = None,
    ):
        """Sample frames from input tensors.

        Args:
            n_frames_per_video: Number of frames per video to sample. -1 indicates that all frames
                should be sampled.
            training_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during training.
            evaluation_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during evaluation.
            dim: The dimension along which to slice the tensors.
            seed: Random number generator seed to deterministic sampling during evaluation.
            per_epoch: Sampling of frames over epochs, this ensures that after
                n_frames / n_frames_per_video epochs all frames have been seen at least once.
                In the case of uneven division, some frames will be seen more than once.
            shuffle_buffer_size: Size of shuffle buffer used during training. An additional
                shuffling step ensures each batch contains a diverse set of images and not only
                images from the same video.
            n_eval_frames_per_video: Number of frames per video to sample on the evaluation splits.
        """
        self.n_frames_per_video = n_frames_per_video
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self.seed = seed
        self.per_epoch = per_epoch
        self.shuffle_buffer_size = shuffle_buffer_size
        if n_eval_frames_per_video is not None:
            self.n_eval_frames_per_video = n_eval_frames_per_video
        else:
            self.n_eval_frames_per_video = n_frames_per_video

    def slice_data(self, data, index: int):
        """Small utility method to slice a numpy array along a specified axis."""
        n_dims_before = self.dim
        n_dims_after = data.ndim - 1 - self.dim
        slices = (slice(None),) * n_dims_before + (index,) + (slice(None),) * n_dims_after
        return data[slices]

    def sample_frames_using_key(self, data, fields, seed, n_frames_per_video):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        # import ipdb
        # ipdb.set_trace()
        # print("in sample_frames_using_key")
        # print(fields)
        for sample in data:
            # Initialize random number generator dependent on instance key. This should make the
            # sampling process deterministic, which is useful when sampling frames for the
            
            # print(sample.keys())
            # print('sample["image"].shape')
            # print(sample["image"].shape)
            # print('type(sample["image"])')
            # print(type(sample["image"]))
            # print({key:type(value) for key, value in sample.items()})
            # print({key:sample[key].shape for key in fields})
            # for key, value in sample.items():
            #     if  isinstance( value, np.ndarray):
            #         print(key,value.dtype)
            # validation/test data.
            key = sample["__key__"]


            n_frames = sample[fields[0]].shape[self.dim]
            frames_per_video = self.n_frames_per_video if self.n_frames_per_video != -1 else n_frames

            if self.per_epoch and self.n_frames_per_video != -1:
                n_different_epochs_per_seed = int(math.ceil(n_frames / frames_per_video))
                try:
                    epoch = int(os.environ["WDS_EPOCH"])
                except KeyError:
                    raise RuntimeError(
                        "Using SampleFramesFromVideo with stratify=True "
                        "requires `WDS_EPOCH` to be set."
                    )
                # Only update the seed after n_frames / n_frames_per_video epochs.
                # This ensures that we get the same random order of frames until
                # we have sampled all of them.
                rand = np.random.RandomState(
                    int(key) + seed + (epoch // n_different_epochs_per_seed)
                )
                indices = rand.permutation(n_frames)
                selected_frames = indices[
                    epoch * self.n_frames_per_video : (epoch + 1) * self.n_frames_per_video
                ].tolist()
                if len(selected_frames) < self.n_frames_per_video:
                    # Input cannot be evenly split, take some frames from the first batch of frames.
                    n_missing = self.n_frames_per_video - len(selected_frames)
                    selected_frames.extend(indices[0:n_missing].tolist())
            else:
                rand = random.Random(int(key) + seed)
                selected_frames = rand.sample(range(n_frames), k=frames_per_video)

            for frame in selected_frames:
                # Slice the fields according to the frame, we use copy in order to allow freeing of
                # the original tensor.
                sliced_fields = {
                    field: self.slice_data(sample[field], frame).copy() for field in fields
                }
                # Leave all fields besides the sliced ones as before, augment the __key__ field to
                # include the frame number.
                sliced_fields["__key__"] = f"{key}_{frame}"
                to_return = {**sample, **sliced_fields}
                # for key, value in to_return.items():
                #     if isinstance(value, np.ndarray):
                #         print(key, value.shape)
                # yield to_return
                yield {**sample, **sliced_fields}

            # Delete fields to be sure we remove all references.
            for field in fields:
                del sample[field]

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key,
                        fields=self._training_fields,
                        seed=self.seed,
                        n_frames_per_video=self.n_frames_per_video,
                    )
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key,
                        fields=self._evaluation_fields,
                        seed=self.seed + 1,
                        n_frames_per_video=self.n_eval_frames_per_video,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling


class SplitConsecutiveFrames(Plugin):
    def __init__(
        self,
        n_consecutive_frames: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
        shuffle_buffer_size: int = 1000,
        drop_last: bool = True,
    ):
        self.n_consecutive_frames = n_consecutive_frames
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self.shuffle_buffer_size = shuffle_buffer_size
        self.drop_last = drop_last

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    def split_to_consecutive_frames(self, data, fields):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[fields[0]].shape[self.dim]

            splitted_fields = [
                np.array_split(
                    sample[field],
                    range(self.n_consecutive_frames, n_frames, self.n_consecutive_frames),
                    axis=self.dim,
                )
                for field in fields
            ]

            for i, slices in enumerate(zip(*splitted_fields)):
                if self.drop_last and slices[0].shape[self.dim] < self.n_consecutive_frames:
                    # Last slice of not equally divisible input, discard.
                    continue

                sliced_fields = dict(zip(fields, slices))
                sliced_fields["__key__"] = f"{key}_{i}"
                yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.split_to_consecutive_frames, fields=self._training_fields)
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.split_to_consecutive_frames,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling


class VideoDecoder(Plugin):
    """Video decoder based on torchaudio StreamReader."""

    def __init__(
        self,
        input_fields: Union[List[str], str],
        stride: int = 1,
        split_extension: bool = True,
        video_reader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Video decoder based on decord.

        It will decode the whole video into a single tensor and can be used with other downstream
        processing plugins.

        Args:
            input_fields (str): The field of the input dictionary containing the video bytes.
            stride (int): Downsample frames by using striding. Default: 1
            split_extension (bool): Split the extension off the field name.
            video_reader_kwargs (Dict[str, Any]): Arguments to decord.VideoReader.
        """
        self.input_fields = list(input_fields) if isinstance(input_fields, list) else [input_fields]
        self.stride = stride
        self.split_extension = split_extension
        self.video_reader_kwargs = video_reader_kwargs if video_reader_kwargs else {}

    def _chunk_iterator(
        self, vrs: Mapping[str, decord.VideoReader], key: str, inputs: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor]:
        """Iterate over chunks of the video.

        For the video decoder we simply return a single chunk containing the whole video, subclasses
        might override this method though.

        Returns:
            str: Derived key which combines chunk and video key.
            torch.Tensor: Chunk of video data.
            Dict: Additional information, for example which frames where selected.  This might be of
                relevance when different modalities need to be sliced in a similar fashion as the
                video input.
        """
        # Get whole video.
        indices = list(range(0, len(next(iter(vrs.values()))), self.stride))
        videos = {output_name: vr.get_batch(indices) for output_name, vr in vrs.items()}
        yield key, {**videos, "decoded_indices": indices}

    @hooks.hook_implementation
    def training_fields(self):
        return self.input_fields

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self.input_fields

    def video_decoding(self, input_generator, chunking):
        for input_data in input_generator:
            key = input_data["__key__"]
            vrs = {}
            for input_field in self.input_fields:
                video_bytes: bytes = input_data[input_field]
                if self.split_extension:
                    output_field, _ = os.path.splitext(input_field)
                else:
                    output_field = input_field
                # Remove the input field
                del input_data[input_field]
                with BytesIO(video_bytes) as f:
                    # We can directly close the file again as VideoReader makes an internal copy.
                    vr = decord.VideoReader(f, **self.video_reader_kwargs)
                vrs[output_field] = vr

            for derived_key, videos_and_additional_info in chunking(vrs, key, input_data):
                yield {
                    **input_data,
                    "__key__": derived_key,
                    **videos_and_additional_info,
                }

    @hooks.hook_implementation
    def training_transform(self):
        return lambda pipeline: pipeline.then(
            functools.partial(self.video_decoding, chunking=self._chunk_iterator)
        )

    @hooks.hook_implementation
    def evaluation_transform(self):
        return lambda pipeline: pipeline.then(
            functools.partial(self.video_decoding, chunking=self._chunk_iterator)
        )


class DecodeRandomWindow(VideoDecoder):
    """Decode a random window of the video."""

    def __init__(self, n_consecutive_frames: int, **video_decoder_args):
        self.n_consecutive_frames = n_consecutive_frames
        self._random = None
        super().__init__(**video_decoder_args)

    @property
    def random(self):
        if not self._random:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                self._random = random.Random(worker_info.seed)
            else:
                self._random = random.Random(torch.initial_seed())

        return self._random

    def _chunk_iterator(
        self, vrs: Mapping[str, decord.VideoReader], key: str, inputs: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor]:
        """Iterate over chunks of the video.

        Returns:
            str: Derived key which combines chunk and video key.
            torch.Tensor: Chunk of video data.
            Dict: Additional information, for example which frames where selected.  This might be of
                relevance when different modalities need to be sliced in a similar fashion as the
                video input.
        """
        n_frames = len(next(iter(vrs.values())))
        assert self.n_consecutive_frames * self.stride < n_frames
        starting_index = self.random.randint(0, n_frames - self.n_consecutive_frames * self.stride)
        indices = list(
            range(
                starting_index, starting_index + self.n_consecutive_frames * self.stride, self.stride
            )
        )
        videos = {output_field: vr.get_batch(indices) for output_field, vr in vrs.items()}
        yield f"{key}_{starting_index}", {**videos, "decoded_indices": indices}

    @hooks.hook_implementation
    def evaluation_transform(self):
        # Do not split during evaluation.
        return lambda pipeline: pipeline.then(
            functools.partial(
                self.video_decoding, chunking=functools.partial(VideoDecoder._chunk_iterator, self)
            )
        )


class DecodeRandomStridedWindow(DecodeRandomWindow):
    """Decode random strided segment of input video."""

    def _chunk_iterator(
        self, vrs: Mapping[str, decord.VideoReader], key: str, inputs: Dict[str, Any]
    ) -> Tuple[str, torch.Tensor]:
        """Iterate over chunks of the video.

        For the video decoder we simply return a single chunk containing the whole video, subclasses
        might override this method though.

        Returns:
            str: Derived key which combines chunk and video key.
            torch.Tensor: Chunk of video data.
            Dict: Additional information, for example which frames where selected.  This might be of
                relevance when different modalities need to be sliced in a similar fashion as the
                video input.
        """
        n_frames = len(next(iter(vrs.values())))
        segment_indices = list(range(0, n_frames + 1, self.n_consecutive_frames * self.stride))
        segment_index = self.random.randint(0, len(segment_indices) - 2)
        indices = list(
            range(segment_indices[segment_index], segment_indices[segment_index + 1], self.stride)
        )
        videos = {output_field: vr.get_batch(indices) for output_field, vr in vrs.items()}
        yield f"{key}_{segment_index}", {**videos, "decoded_indices": indices}


class RandomStridedWindow(Plugin):
    """Select a random consecutive subsequence of frames in a strided manner.

    Given a sequence of [1, 2, 3, 4, 5, 6, 7, 8, 9] this will return one of
    [1, 2, 3] [4, 5, 6] [7, 8, 9].
    """

    def __init__(
        self,
        n_consecutive_frames: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
    ):
        self.n_consecutive_frames = n_consecutive_frames
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self._random = None

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @property
    def random(self):
        if not self._random:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info:
                self._random = random.Random(worker_info.seed)
            else:
                self._random = random.Random(torch.initial_seed())

        return self._random

    def split_to_consecutive_frames(self, data, fields):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[fields[0]].shape[self.dim]

            splitted_fields = [
                np.array_split(
                    sample[field],
                    range(self.n_consecutive_frames, n_frames, self.n_consecutive_frames),
                    axis=self.dim,
                )
                for field in fields
            ]

            n_fragments = len(splitted_fields[0])

            if len(splitted_fields[0][-1] < self.n_consecutive_frames):
                # Discard last fragment if too short.
                n_fragments -= 1

            fragment_id = self.random.randint(0, n_fragments - 1)
            sliced_fields = {
                field_name: splitted_field[fragment_id]
                for field_name, splitted_field in zip(fields, splitted_fields)
            }
            sliced_fields["__key__"] = f"{key}_{fragment_id}"
            yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.split_to_consecutive_frames, fields=self._training_fields)
                )
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.split_to_consecutive_frames,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling


def rename_according_to_mapping(input: dict, mapping: dict):
    # print("In raname")
    output = {key: value for key, value in input.items() if key not in mapping.keys()}
    for source, target in mapping.items():
        output[target] = input[source]
        # if isinstance(output[target], np.ndarray):
        #     print(target, output[target].shape)
    return output


class RenameFields(Plugin):
    def __init__(
        self,
        train_mapping: Optional[Dict[str, str]] = None,
        evaluation_mapping: Optional[Dict[str, str]] = None,
    ):
        super().__init__()
        self.train_mapping = train_mapping if train_mapping else {}
        self.evaluation_mapping = evaluation_mapping if evaluation_mapping else {}

    @hooks.hook_implementation
    def training_fields(self) -> Tuple[str]:
        return tuple(self.train_mapping.keys())

    @hooks.hook_implementation
    def training_transform(self):
        def rename_fields(pipeline: webdataset.Processor):
            if len(self.training_fields()):
                return pipeline.map(
                    functools.partial(rename_according_to_mapping, mapping=self.train_mapping)
                )
            else:
                return pipeline

        return rename_fields

    # Do same thing during training and testing.
    @hooks.hook_implementation
    def evaluation_fields(self) -> Tuple[str]:
        return tuple(self.evaluation_mapping.keys())

    @hooks.hook_implementation
    def evaluation_transform(self):
        def rename_fields(pipeline: webdataset.Processor):
            if len(self.evaluation_fields()):
                return pipeline.map(
                    functools.partial(rename_according_to_mapping, mapping=self.evaluation_mapping)
                )
            else:
                return pipeline

        return rename_fields


class DeterministicSubsampleWithMasking(Plugin):
    def __init__(
        self,
        samples_per_instance: int,
        training_fields: Optional[List[str]] = None,
        evaluation_fields: Optional[List[str]] = None,
        mask_field: Optional[str] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.samples_per_instance = samples_per_instance
        self._training_fields = training_fields if training_fields else []
        self._evaluation_fields = evaluation_fields if evaluation_fields else []
        self.mask_field = mask_field
        self.seed = seed

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    def subsample_with_masking(self, instance, fields):
        key = instance["__key__"]
        random_state = random.Random(int(key) + self.seed)
        n_frames = instance[fields[0]].shape[0]
        indices = np.array(random_state.sample(range(n_frames), self.samples_per_instance))

        output = instance.copy()
        for field in fields:
            values_to_keep = instance[field][indices]
            field_output = np.full_like(instance[field], np.NaN)
            field_output[indices] = values_to_keep
            output[field] = field_output

        if self.mask_field:
            mask = np.zeros(n_frames, dtype=bool)
            mask[indices] = True
            output[self.mask_field] = mask

        return output

    @hooks.hook_implementation
    def training_transform(self):
        def subsample_with_masking(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.map(
                    functools.partial(self.subsample_with_masking, fields=self._training_fields)
                )
            else:
                return pipeline

        return subsample_with_masking

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def subsample_with_masking(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.map(
                    functools.partial(
                        self.subsample_with_masking,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return subsample_with_masking


class SpatialSlidingWindow(Plugin):
    """Split image data spatially by sliding a window across."""

    def __init__(
        self,
        window_size: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int, int, int],
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        expected_n_windows: Optional[int] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.padding = padding
        self.expected_n_windows = expected_n_windows
        self._training_fields = tuple(training_fields) if training_fields else tuple()
        self._evaluation_fields = tuple(evaluation_fields) if evaluation_fields else tuple()

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @staticmethod
    def pad(elem, padding):
        if elem.shape[-1] != 1 and elem.shape[-1] != 3:
            elem = elem[..., None]
        orig_height = elem.shape[-3]
        orig_width = elem.shape[-2]

        p_left, p_top, p_right, p_bottom = padding
        height = orig_height + p_top + p_bottom
        width = orig_width + p_left + p_right

        padded_shape = list(elem.shape[:-3]) + [height, width, elem.shape[-1]]
        elem_padded = np.zeros_like(elem, shape=padded_shape)
        elem_padded[..., p_top : p_top + orig_height, p_left : p_left + orig_width, :] = elem

        return elem_padded

    def sliding_window(self, data, fields):
        for sample in data:
            key = sample["__key__"]

            window_x, window_y = self.window_size
            stride_x, stride_y = self.stride
            padded_elems = {key: self.pad(sample[key], self.padding) for key in fields}

            n_windows = 0
            x = 0
            y = 0
            while True:
                shape = None
                windowed_fields = {}
                for key in fields:
                    elem_padded = padded_elems[key]
                    if shape is None:
                        shape = elem_padded.shape
                    else:
                        if shape[-3:-1] != elem_padded.shape[-3:-1]:
                            raise ValueError("Element height, width after padding do not match")
                    windowed_fields[key] = elem_padded[..., y : y + window_y, x : x + window_x, :]

                    window_height, window_width = windowed_fields[key].shape[-3:-1]
                    assert (
                        window_y == window_height and window_x == window_width
                    ), f"Expected {window_y}, {window_x}, received {window_height}, {window_width}"

                windowed_fields["__key__"] = f"{key}_{x - self.padding[0]}_{y - self.padding[1]}"
                yield {**sample, **windowed_fields}
                n_windows += 1

                x += stride_x
                if x >= shape[-2]:
                    y += stride_y
                    x = 0
                if y >= shape[-3]:
                    break

            if self.expected_n_windows is not None and self.expected_n_windows != n_windows:
                raise ValueError(f"Expected {self.expected_n_windows} windows, but got {n_windows}")

    @hooks.hook_implementation
    def training_transform(self):
        def apply_sliding_window(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.sliding_window, fields=self._training_fields)
                )
            else:
                return pipeline

        return apply_sliding_window

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_sliding_window(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sliding_window,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return apply_sliding_window


class MaskInstances(Plugin):
    """Filter instances by masking non matching with NaN."""

    def __init__(
        self,
        training_fields: Optional[List[str]] = None,
        training_keys_to_keep: Optional[List[str]] = None,
        evaluation_fields: Optional[List[str]] = None,
        evaluation_keys_to_keep: Optional[List[str]] = None,
        mask_video: bool = False,
    ):
        self._training_fields = training_fields
        self.training_keys_to_keep = set(training_keys_to_keep) if training_keys_to_keep else None
        self._evaluation_fields = evaluation_fields
        self.evaluation_keys_to_keep = (
            set(evaluation_keys_to_keep) if evaluation_keys_to_keep else None
        )
        self.mask_video = mask_video
        if self.mask_video:
            if self.training_keys_to_keep is not None:
                self.train_video_key_to_frame_mapping = defaultdict(set)
                for key in self.training_keys_to_keep:
                    video_key, frame = key.split("_")
                    self.train_video_key_to_frame_mapping[video_key].add(int(frame))
            if self.evaluation_keys_to_keep is not None:
                self.eval_video_key_to_frame_mapping = defaultdict(list)
                for key in self.evaluation_keys_to_keep:
                    video_key, frame = key.split("_")
                    self.eval_video_key_to_frame_mapping[video_key].add(int(frame))

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    def mask_instance(self, instance, fields, keys):
        key = instance["__key__"]

        if key not in keys:
            for field in fields:
                data = instance[field]
                if isinstance(data, np.ndarray):
                    instance[field] = np.full_like(data, np.NaN)
                elif isinstance(data, torch.Tensor):
                    instance[field] = torch.full_like(data, np.NaN)
                else:
                    raise RuntimeError(f"Field {field} is of unexpected type {type(data)}.")
        return instance

    def mask_instance_video(self, instance, fields, video_key_to_frame_mapping):
        key = instance["__key__"]
        output = instance.copy()
        for field in fields:
            data = instance[field]
            if isinstance(data, np.ndarray):
                output[field] = np.full_like(data, np.NaN)
            elif isinstance(data, torch.Tensor):
                output[field] = torch.full_like(data, np.NaN)
            else:
                raise RuntimeError(f"Field {field} is of unexpected type {type(data)}.")

        # We need to do some special handling here due to the strided decoding.
        # This is not really nice, but fixing it nicely would require significantly
        # more work for which we do not have the time at the moment.
        if "decoded_indices" in instance.keys():
            # Input comes from strided decoding, we thus need to adapt
            # key and frames.
            key, _ = key.split("_")  # Get video key.
            key = str(int(key))
            if key in video_key_to_frame_mapping.keys():
                frames_to_keep = video_key_to_frame_mapping[key]
                decoded_indices = instance["decoded_indices"]
                frames_to_keep = [index for index in decoded_indices if index in frames_to_keep]
                for field in fields:
                    data = instance[field]
                    output[field][frames_to_keep] = data[frames_to_keep]
        else:
            if key in video_key_to_frame_mapping.keys():
                frames_to_keep = video_key_to_frame_mapping[key]
                for field in fields:
                    data = instance[field]
                    output[field][frames_to_keep] = data[frames_to_keep]
        return output

    @hooks.hook_implementation
    def training_transform(self):
        def subsample_with_masking(pipeline: webdataset.Processor):
            if self._training_fields:
                if self.mask_video:
                    return pipeline.map(
                        functools.partial(
                            self.mask_instance_video,
                            fields=self._training_fields,
                            video_key_to_frame_mapping=self.train_video_key_to_frame_mapping,
                        )
                    )
                else:
                    return pipeline.map(
                        functools.partial(
                            self.mask_instance,
                            fields=self._training_fields,
                            keys=self.training_keys_to_keep,
                        )
                    )
            else:
                return pipeline

        return subsample_with_masking

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def subsample_with_masking(pipeline: webdataset.Processor):
            if self._evaluation_fields:
                if self.mask_video:
                    return pipeline.map(
                        functools.partial(
                            self.mask_instance_video,
                            fields=self._evaluation_fields,
                            video_key_to_frame_mapping=self.eval_video_key_to_frame_mapping,
                        )
                    )
                else:
                    return pipeline.map(
                        functools.partial(
                            self.mask_instance,
                            fields=self._evaluation_fields,
                            keys=self.evaluation_keys_to_keep,
                        )
                    )
            else:
                return pipeline

        return subsample_with_masking


class FlattenVideoToImage(Plugin):
    def __init__(
        self,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        shuffle_buffer_size: int = 0,
    ):
        """Flatten input video tensors into images.

        Args:
            training_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during training.
            evaluation_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during evaluation.
            shuffle_buffer_size: Size of shuffle buffer used during training. An additional
                shuffling step ensures each batch contains a diverse set of images and not only
                images from the same video.
        """
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.shuffle_buffer_size = shuffle_buffer_size

    def flatten_video(self, data, fields):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            # Initialize random number generator dependent on instance key. This should make the
            # sampling process deterministic, which is useful when sampling frames for the
            # validation/test data.
            key = sample["__key__"]
            # TODO (hornmax): We assume all fields to have the same size. I do not want to check
            # this here as it seems a bit verbose.
            n_frames = sample[fields[0]].shape[0]

            for frame in range(n_frames):
                # Slice the fields according to the frame.
                sliced_fields = {field: sample[field][frame] for field in fields}
                # Leave all fields besides the sliced ones as before, augment the __key__ field to
                # include the frame number.
                sliced_fields["__key__"] = f"{key}_{frame}"
                yield {**sample, **sliced_fields}

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        def flatten_video(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(self.flatten_video, fields=self._training_fields)
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return flatten_video

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def flatten_video(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.flatten_video,
                        fields=self._evaluation_fields,
                    )
                )
            else:
                return pipeline

        return flatten_video

def transform_lift_dict(elements: dict, *, transform_dict, element_key: str, element_key_to_lift: List[str], to_del):
    """Utility function to fix issues with pickling."""
    # print("In transform_lift_dict")
    if not to_del:
        for key in element_key_to_lift:
            # print("before transform")
            # print(key, elements[element_key][key].shape)
            elements[key] = transform_dict[key](elements[element_key][key])
            # if isinstance(elements[key], np.ndarray):
            #     print("after transform")
            #     print(key, elements[key].shape)
        return elements
    else:
        for key in element_key_to_lift:
            # print("before transform")
            # print(key, elements[element_key][key].shape)
            elements[key] = transform_dict[key](elements[element_key][key])
            # if isinstance(elements[key], np.ndarray):
            #     print("after transform")
            #     print(key, elements[key].shape)
        elements.pop(element_key, None) 
        # print("element_key.keys() in transform_lift_dict")
        # print(element_key.keys())
        # import ipdb
        # ipdb.set_trace()
        return elements     

class SingleElementPreprocessingLiftDict(Plugin):
    """Preprocessing of a single element in the input data.

    This is useful to build preprocessing pipelines based on existing element transformations such
    as those provided by torchvision. The element can optionally be duplicated and stored under a
    different key after the transformation by specifying `duplicate_key`. This is useful to further
    preprocess this element in different ways afterwards.
    """

    def __init__(
        self,
        training_transform: Dict[str, Callable],
        evaluation_transform: Dict[str, Callable],
        element_key: str = "image",
        element_key_to_lift: List[str] = None,
        to_del = False
    ):
        
        self._training_transform = training_transform
        self._evaluation_transform = evaluation_transform
        self.element_key = element_key
        self.element_key_to_lift = element_key_to_lift
        self.to_del = to_del
 

    @hooks.hook_implementation
    def training_fields(self):
        return (self.element_key, )


    
    @hooks.hook_implementation
    def training_transform(self):
        if self._training_transform:
            # print("self._training_transform")
            # for key, value in self._training_transform.items():
            #     print(key, value)
            def transform(pipeline: webdataset.Processor):
                transform_func = functools.partial(
                    transform_lift_dict,
                    transform_dict=self._training_transform,
                    element_key=self.element_key,
                    element_key_to_lift=self.element_key_to_lift,
                    to_del = self.to_del
                )
                return pipeline.map(transform_func)
            return transform
        else:
            return None

    @hooks.hook_implementation
    def evaluation_fields(self):
        return (self.element_key, )
    
    @hooks.hook_implementation
    def evaluation_transform(self):

        if self._evaluation_transform:
            # print("self._evaluation_transform")
            # for key, value in self._evaluation_transform.items():
            #     print(key, value)
            def transform(pipeline: webdataset.Processor):
                transform_func = functools.partial(
                    transform_lift_dict,
                    transform_dict=self._evaluation_transform,
                    element_key=self.element_key,
                    element_key_to_lift=self.element_key_to_lift,
                    to_del = self.to_del
                )
                return pipeline.map(transform_func)
            return transform
        else:
            return None

class SequenceSampleFramesFromVideo(Plugin):
    def __init__(
        self,
        n_frames_per_video: int,
        training_fields: Sequence[str],
        evaluation_fields: Sequence[str],
        dim: int = 0,
        seed: int = 39480234,
        per_epoch: bool = False,
        shuffle_buffer_size: int = 1000,
        n_eval_frames_per_video: Optional[int] = None,
    ):
        """Sample frames from input tensors.

        Args:
            n_frames_per_video: Number of frames per video to sample. -1 indicates that all frames
                should be sampled.
            training_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during training.
            evaluation_fields: The fields that should be considered video data and thus sliced
                according to the frame sampling during evaluation.
            dim: The dimension along which to slice the tensors.
            seed: Random number generator seed to deterministic sampling during evaluation.
            per_epoch: Sampling of frames over epochs, this ensures that after
                n_frames / n_frames_per_video epochs all frames have been seen at least once.
                In the case of uneven division, some frames will be seen more than once.
            shuffle_buffer_size: Size of shuffle buffer used during training. An additional
                shuffling step ensures each batch contains a diverse set of images and not only
                images from the same video.
            n_eval_frames_per_video: Number of frames per video to sample on the evaluation splits.
        """
        self.n_frames_per_video = n_frames_per_video
        self._training_fields = tuple(training_fields)
        self._evaluation_fields = tuple(evaluation_fields)
        self.dim = dim
        self.seed = seed
        self.per_epoch = per_epoch
        self.shuffle_buffer_size = shuffle_buffer_size
        if n_eval_frames_per_video is not None:
            self.n_eval_frames_per_video = n_eval_frames_per_video
        else:
            self.n_eval_frames_per_video = n_frames_per_video

    def slice_data(self, data, index: int):
        """Small utility method to slice a numpy array along a specified axis."""
        n_dims_before = self.dim
        n_dims_after = data.ndim - 1 - self.dim
        slices = (slice(None),) * n_dims_before + (index,) + (slice(None),) * n_dims_after
        return data[slices]

    def sample_frames_using_key(self, data, fields, seed, n_frames_per_video):
        """Sample frames deterministically from generator of videos using the __key__ field."""
        for sample in data:
            key = sample["__key__"]
            n_frames = sample[fields[0]].shape[self.dim]

            splitted_fields = [
                np.array_split(
                    sample[field],
                    range(self.n_consecutive_frames, n_frames, self.n_consecutive_frames),
                    axis=self.dim,
                )
                for field in fields
            ]

            for i, slices in enumerate(zip(*splitted_fields)):
                if self.drop_last and slices[0].shape[self.dim] < self.n_consecutive_frames:
                    # Last slice of not equally divisible input, discard.
                    continue

                sliced_fields = dict(zip(fields, slices))
                sliced_fields["__key__"] = f"{key}_{i}"
                yield {**sample, **sliced_fields}
        
        for sample in data:
            key = sample["__key__"]


            n_frames = sample[fields[0]].shape[self.dim]
            frames_per_video = self.n_frames_per_video if self.n_frames_per_video != -1 else n_frames
            slices_id_start = range(0, n_frames, self.n_consecutive_frames)

            if self.per_epoch and self.n_frames_per_video != -1:
                n_different_epochs_per_seed = int(math.ceil(n_frames / frames_per_video))
                try:
                    epoch = int(os.environ["WDS_EPOCH"])
                except KeyError:
                    raise RuntimeError(
                        "Using SampleFramesFromVideo with stratify=True "
                        "requires `WDS_EPOCH` to be set."
                    )
                # Only update the seed after n_frames / n_frames_per_video epochs.
                # This ensures that we get the same random order of frames until
                # we have sampled all of them.
                rand = np.random.RandomState(
                    int(key) + seed + (epoch // n_different_epochs_per_seed)
                )
                indices = rand.permutation(n_frames)
                selected_frames = indices[
                    epoch * self.n_frames_per_video : (epoch + 1) * self.n_frames_per_video
                ].tolist()
                if len(selected_frames) < self.n_frames_per_video:
                    # Input cannot be evenly split, take some frames from the first batch of frames.
                    n_missing = self.n_frames_per_video - len(selected_frames)
                    selected_frames.extend(indices[0:n_missing].tolist())
            else:
                rand = random.Random(int(key) + seed)
                selected_frames = rand.sample(range(n_frames), k=frames_per_video)

            for frame in selected_frames:
                # Slice the fields according to the frame, we use copy in order to allow freeing of
                # the original tensor.
                sliced_fields = {
                    field: self.slice_data(sample[field], frame).copy() for field in fields
                }
                # Leave all fields besides the sliced ones as before, augment the __key__ field to
                # include the frame number.
                sliced_fields["__key__"] = f"{key}_{frame}"
                yield {**sample, **sliced_fields}

            # Delete fields to be sure we remove all references.
            for field in fields:
                del sample[field]

    @hooks.hook_implementation
    def training_fields(self):
        return self._training_fields

    @hooks.hook_implementation
    def training_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._training_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key,
                        fields=self._training_fields,
                        seed=self.seed,
                        n_frames_per_video=self.n_frames_per_video,
                    )
                ).shuffle(self.shuffle_buffer_size)
            else:
                return pipeline

        return apply_deterministic_sampling

    @hooks.hook_implementation
    def evaluation_fields(self):
        return self._evaluation_fields

    @hooks.hook_implementation
    def evaluation_transform(self):
        def apply_deterministic_sampling(pipeline: webdataset.Processor):
            if len(self._evaluation_fields) > 0:
                return pipeline.then(
                    functools.partial(
                        self.sample_frames_using_key,
                        fields=self._evaluation_fields,
                        seed=self.seed + 1,
                        n_frames_per_video=self.n_eval_frames_per_video,
                    )
                )
            else:
                return pipeline

        return apply_deterministic_sampling
