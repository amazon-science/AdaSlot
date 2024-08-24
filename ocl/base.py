import abc
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import pluggy
import torch
from torch import nn
from torchtyping import TensorType

PluggyHookRelay = pluggy._hooks._HookRelay  # Type alias for more readable function signatures

ConditioningOutput = TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821


class Conditioning(nn.Module, metaclass=abc.ABCMeta):
    """Base class for conditioning perceptual grouping."""

    @abc.abstractmethod
    def forward(self, *args) -> ConditioningOutput:
        pass


@dataclasses.dataclass
class FrameFeatures:
    """Features associated with a single frame."""

    features: TensorType["batch_size", "n_spatial_features", "feature_dim"]  # noqa: F821
    positions: TensorType["n_spatial_features", "spatial_dims"]  # noqa: F821


@dataclasses.dataclass
class FeatureExtractorOutput:
    """Output of feature extractor."""

    features: TensorType["batch_size", "frames", "n_spatial_features", "feature_dim"]  # noqa: F821
    positions: TensorType["n_spatial_features", "spatial_dims"]  # noqa: F821
    aux_features: Optional[Dict[str, torch.Tensor]] = None

    def __iter__(self):
        """Iterate over features and positions per frame."""
        for frame_features in torch.split(self.features, 1, dim=1):
            yield FrameFeatures(frame_features.squeeze(1), self.positions)


class FeatureExtractor(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class for Feature Extractors.

    We expect that the forward method returns a flattened representation of the features, to make
    outputs consistent and not dependent on equal spacing or the dimensionality of the spatial
    information.
    """

    @property
    @abc.abstractmethod
    def feature_dim(self):
        """Get dimensionality of the features.

        Returns:
            int: The dimensionality of the features.
        """

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor) -> FeatureExtractorOutput:
        pass


@dataclasses.dataclass
class PerceptualGroupingOutput:
    """Output of a perceptual grouping algorithm."""

    objects: TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
    is_empty: Optional[TensorType["batch_size", "n_objects"]] = None  # noqa: F821
    feature_attributions: Optional[
        TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821
    ] = None


class PerceptualGrouping(nn.Module, metaclass=abc.ABCMeta):
    """Abstract base class of a perceptual grouping algorithm."""

    @abc.abstractmethod
    def forward(self, extracted_features: FeatureExtractorOutput) -> PerceptualGroupingOutput:
        pass

    @property
    @abc.abstractmethod
    def object_dim(self):
        pass


class Instances:
    """Modified from Detectron2 (https://github.com/facebookresearch/detectron2).

    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.

    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.

    Some basic usage:

    1. Set/get/check a field:

       .. code-block:: python

          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)

    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``

       .. code-block:: python

          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """Init function.

        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """Set the field named `name` to `value`.

        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert (
                len(self) == data_len
            ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """Returns whether the field called `name` exists."""
        return name in self._fields

    def remove(self, name: str) -> None:
        """Remove the field called `name`."""
        del self._fields[name]

    def get(self, name: str) -> Any:
        """Returns the field called `name`."""
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """Get field.

        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """To device.

        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """Get entry.

        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """Concatenate instances.

        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__
