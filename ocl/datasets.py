"""Implementation of datasets."""
import collections
import logging
import math
import os
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import webdataset
from torch.utils.data._utils import collate as torch_collate

from ocl import base
from ocl.hooks import FakeHooks, hook_implementation

LOGGER = logging.getLogger(__name__)


def filter_keys(d: dict, keys_to_keep=tuple):
    """Filter dict for keys in keys_to_keep."""
    # print("keys_to_keep")
    # print(keys_to_keep)
    keys_to_keep = ("_",) + keys_to_keep
    return {
        key: value
        for key, value in d.items()
        if any(key.startswith(prefix) for prefix in keys_to_keep)
    }


def combine_keys(list_of_key_tuples: Sequence[Tuple[str]]):
    return tuple(set(chain.from_iterable(list_of_key_tuples)))


class WebdatasetDataModule(pl.LightningDataModule):
    """Imagenet Data Module."""

    def __init__(
        self,
        train_shards: Optional[Union[str, List[str]]] = None,
        val_shards: Optional[Union[str, List[str]]] = None,
        test_shards: Optional[Union[str, List[str]]] = None,
        batch_size: int = 32,
        eval_batch_size: Optional[int] = None,
        num_workers: int = 2,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        shuffle_train: bool = True,
        shuffle_buffer_size: int = 3000,
        use_autopadding: bool = False,
        continue_on_shard_error: bool = False,
        hooks: Optional[base.PluggyHookRelay] = None,
    ):
        super().__init__()
        if train_shards is None and val_shards is None and test_shards is None:
            raise ValueError("No split was specified. Need to specify at least one split.")
        self.train_shards = train_shards
        self.val_shards = val_shards
        self.test_shards = test_shards
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_buffer_size = shuffle_buffer_size
        self.continue_on_shard_error = continue_on_shard_error
        self.hooks = hooks if hooks else FakeHooks()

        if use_autopadding:
            self.collate_fn = collate_with_autopadding
        else:
            self.collate_fn = collate_with_batch_size

    @staticmethod
    def _remove_extensions(input_dict):
        def _remove_extension(name: str):
            if name.endswith(".gz"):
                # Webdataset automatically decompresses these, we want to remove two layers of
                # extensions due to that.
                name = os.path.splitext(name)[0]
            return os.path.splitext(name)[0]

        return {_remove_extension(name): value for name, value in input_dict.items()}

    @hook_implementation
    def on_train_epoch_start(self, model) -> None:
        """Set environment variables required for better shuffling."""
        # Required for shuffling of instances across workers, see `epoch_shuffle` parameter of
        # `webdataset.PytorchShardList`.
        os.environ["WDS_EPOCH"] = str(model.current_epoch)

    def _create_webdataset(
        self,
        uri_expression: Union[str, List[str]],
        shuffle=False,
        n_datapoints: Optional[int] = None,
        keys_to_keep: Sequence[str] = tuple(),
        transforms: Sequence[Callable[[webdataset.Processor], webdataset.Processor]] = tuple(),
    ):
        shard_list = webdataset.PytorchShardList(
            uri_expression, shuffle=shuffle, epoch_shuffle=shuffle
        )

        if self.continue_on_shard_error:
            handler = webdataset.warn_and_continue
        else:
            handler = webdataset.reraise_exception
        dataset = webdataset.WebDataset(shard_list, handler=handler)
        # Discard unneeded properties of the elements prior to shuffling and decoding.
        dataset = dataset.map(partial(filter_keys, keys_to_keep=keys_to_keep))

        if shuffle:
            dataset = dataset.shuffle(self.shuffle_buffer_size)

        # Decode files and remove extensions from input as we already decoded the elements. This
        # makes our pipeline invariant to the exact encoding used in the dataset.
        dataset = dataset.decode("rgb8").map(WebdatasetDataModule._remove_extensions)

        # Apply transforms
        for transform in transforms:
            dataset = transform(dataset)
        return dataset.with_length(n_datapoints)

    def _create_dataloader(self, dataset, batch_transforms, size, batch_size, partial_batches):
        # Don't return partial batches during training as these give the partial samples a higher
        # weight in the optimization than the other samples of the dataset.

        # Apply batch transforms.
        dataset = dataset.batched(
            batch_size,
            partial=partial_batches,
            collation_fn=self.collate_fn,
        )
        for transform in batch_transforms:
            dataset = transform(dataset)
        dataloader = webdataset.WebLoader(
            dataset,
            num_workers=self.num_workers,
            batch_size=None,
        )

        if size:
            # This is required for ddp training as we otherwise cannot guarantee that each worker
            # gets the same number of batches.
            equalized_size: int
            if partial_batches:
                # Round up in the case of partial batches.
                equalized_size = int(math.ceil(size / batch_size))
            else:
                equalized_size = size // batch_size

            dataloader = dataloader.ddp_equalize(equalized_size, with_length=True)
        else:
            LOGGER.warning(
                "Size not provided in the construction of webdataset. "
                "This may lead to problems when running distributed training."
            )
        return dataloader

    def train_data_iterator(self):
        if self.train_shards is None:
            raise ValueError("Can not create train_data_iterator. No training split was specified.")
        return self._create_webdataset(
            self.train_shards,
            shuffle=self.shuffle_train,
            n_datapoints=self.train_size,
            keys_to_keep=combine_keys(self.hooks.training_fields()),
            transforms=self.hooks.training_transform(),
        )

    def train_dataloader(self):
        return self._create_dataloader(
            self.train_data_iterator(),
            self.hooks.training_batch_transform(),
            self.train_size,
            self.batch_size,
            partial_batches=False,
        )

    def val_data_iterator(self):
        if self.val_shards is None:
            raise ValueError("Can not create val_data_iterator. No val split was specified.")
        return self._create_webdataset(
            self.val_shards,
            shuffle=False,
            n_datapoints=self.val_size,
            keys_to_keep=combine_keys(self.hooks.evaluation_fields()),
            transforms=self.hooks.evaluation_transform(),
        )

    def val_dataloader(self):
        return self._create_dataloader(
            self.val_data_iterator(),
            self.hooks.evaluation_batch_transform(),
            self.val_size,
            self.eval_batch_size,
            partial_batches=True,
        )

    def test_data_iterator(self):
        if self.test_shards is None:
            raise ValueError("Can not create test_data_iterator. No test split was specified.")
        return self._create_webdataset(
            self.test_shards,
            shuffle=False,
            n_datapoints=self.test_size,
            keys_to_keep=combine_keys(self.hooks.evaluation_fields()),
            transforms=self.hooks.evaluation_transform(),
        )

    def test_dataloader(self):
        return self._create_dataloader(
            self.test_data_iterator(),
            self.hooks.evaluation_batch_transform(),
            self.test_size,
            self.eval_batch_size,
            partial_batches=True,
        )


class DummyDataModule(pl.LightningDataModule):
    """Dataset providing dummy data for testing."""

    def __init__(
        self,
        data_shapes: Dict[str, List[int]],
        data_types: Dict[str, str],
        hooks: Optional[base.PluggyHookRelay] = None,
        batch_size: int = 4,
        eval_batch_size: Optional[int] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        # Remaining args needed for compatibility with other datamodules
        train_shards: Optional[str] = None,
        val_shards: Optional[str] = None,
        test_shards: Optional[str] = None,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.data_shapes = {key: list(shape) for key, shape in data_shapes.items()}
        self.data_types = data_types
        self.hooks = hooks if hooks else FakeHooks
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size if eval_batch_size is not None else batch_size

        self.train_size = train_size
        if self.train_size is None:
            self.train_size = 3 * batch_size + 1
        self.val_size = val_size
        if self.val_size is None:
            self.val_size = 2 * batch_size
        self.test_size = test_size
        if self.test_size is None:
            self.test_size = 2 * batch_size

    @staticmethod
    def _get_random_data_for_dtype(dtype: str, shape: List[int]):
        if dtype == "image":
            return np.random.randint(0, 256, size=shape, dtype=np.uint8)
        elif dtype == "binary":
            return np.random.randint(0, 2, size=shape, dtype=bool)
        elif dtype == "uniform":
            return np.random.rand(*shape).astype(np.float32)
        elif dtype.startswith("categorical_"):
            bounds = [int(b) for b in dtype.split("_")[1:]]
            if len(bounds) == 1:
                lower, upper = 0, bounds[0]
            else:
                lower, upper = bounds
            np_dtype = np.uint8 if upper <= 256 else np.uint64
            return np.random.randint(lower, upper, size=shape, dtype=np_dtype)
        elif dtype.startswith("mask"):
            categories = shape[1]
            np_dtype = np.uint8 if categories <= 256 else np.uint64
            slot_per_pixel = np.random.randint(
                0, categories, size=shape[:1] + shape[2:], dtype=np_dtype
            )
            return (
                np.eye(categories)[slot_per_pixel.reshape(-1)]
                .reshape(shape[:1] + shape[2:] + [categories])
                .transpose([0, 4, 1, 2, 3])
            )
        else:
            raise ValueError(f"Unsupported dtype `{dtype}`")

    def _create_dataset(
        self,
        n_datapoints: int,
        transforms: Sequence[Callable[[Any], Any]],
    ):
        class NumpyDataset(torch.utils.data.IterableDataset):
            def __init__(self, data: Dict[str, np.ndarray], size: int):
                super().__init__()
                self.data = data
                self.size = size

            def __iter__(self):
                for i in range(self.size):
                    elem = {key: value[i] for key, value in self.data.items()}
                    elem["__key__"] = str(i)
                    yield elem

        data = {}
        for key, shape in self.data_shapes.items():
            data[key] = self._get_random_data_for_dtype(self.data_types[key], [n_datapoints] + shape)

        dataset = webdataset.Processor(NumpyDataset(data, n_datapoints), lambda x: x)
        for transform in transforms:
            dataset = transform(dataset)

        return dataset

    def _create_dataloader(self, dataset, batch_size):
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0, collate_fn=collate_with_autopadding
        )

    def train_dataloader(self):
        dataset = self._create_dataset(self.train_size, self.hooks.training_transform())
        return self._create_dataloader(dataset, self.batch_size)

    def val_dataloader(self):
        dataset = self._create_dataset(self.val_size, self.hooks.evaluation_transform())
        return self._create_dataloader(dataset, self.eval_batch_size)

    def test_dataloader(self):
        dataset = self._create_dataset(self.test_size, self.hooks.evaluation_transform())
        return self._create_dataloader(dataset, self.eval_batch_size)


def collate_with_batch_size(batch):
    """Call default pytorch collate function yet for dict type input additionally add batch size."""
    if isinstance(batch[0], collections.abc.Mapping):
        out = torch_collate.default_collate(batch)
        out["batch_size"] = len(batch)
        return out
    return torch_collate.default_collate(batch)


def collate_with_autopadding(batch):
    """Collate function that takes a batch of data and stacks it with a batch dimension.

    In contrast to torch's collate function, this function automatically pads tensors of different
    sizes with zeros such that they can be stacked.

    Adapted from https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py.
    """
    elem = batch[0]
    elem_type = type(elem)
    # print(batch[0])
    # print(elem_type)
    if isinstance(elem, torch.Tensor):
        out = None
        # As most tensors will not need padding to be stacked, we first try to stack them normally
        # and pad only if normal padding fails. This avoids explicitly checking whether all tensors
        # have the same shape before stacking.
        try:
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch)
                if len(batch) * elem.numel() != numel:
                    # Check whether resizing will fail because tensors have unequal sizes to avoid
                    # a memory allocation. This is a sufficient but not necessary condition, so it
                    # can happen that this check will not trigger when padding is necessary.
                    raise RuntimeError()
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch), *elem.shape)
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            # Stacking did not work. Try to pad tensors to the same dimensionality.
            if not all(x.ndim == elem.ndim for x in batch):
                raise ValueError("Tensors in batch have different number of dimensions.")

            shapes = [x.shape for x in batch]
            max_dims = [max(shape[idx] for shape in shapes) for idx in range(elem.ndim)]

            paddings = []
            for shape in shapes:
                padding = []
                # torch.nn.functional.pad wants padding from last to first dim, so go in reverse
                for idx in reversed(range(len(shape))):
                    padding.append(0)
                    padding.append(max_dims[idx] - shape[idx])
                paddings.append(padding)

            batch_padded = [
                torch.nn.functional.pad(x, pad, mode="constant", value=0.0)
                for x, pad in zip(batch, paddings)
            ]

            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum(x.numel() for x in batch_padded)
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage).resize_(len(batch_padded), *batch_padded[0].shape)
            return torch.stack(batch_padded, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if torch_collate.np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(torch_collate.default_collate_err_msg_format.format(elem.dtype))

            return collate_with_autopadding([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, str):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        out = {key: collate_with_autopadding([d[key] for d in batch]) for key in elem}
        out["batch_size"] = len(batch)
        try:
            return elem_type(out)
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return out
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_with_autopadding(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        if isinstance(elem, tuple):
            return [
                collate_with_autopadding(samples) for samples in transposed
            ]  # Backwards compatibility.
        else:
            try:
                return elem_type([collate_with_autopadding(samples) for samples in transposed])
            except TypeError:
                # The sequence type may not support `__init__(iterable)` (e.g., `range`).
                return [collate_with_autopadding(samples) for samples in transposed]

    raise TypeError(torch_collate.default_collate_err_msg_format.format(elem_type))
