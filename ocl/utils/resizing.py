"""Utilities related to resizing of tensors."""
import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ocl.utils.routing import RoutableMixin


class Resize(nn.Module, RoutableMixin):
    """Module resizing tensors."""

    MODES = {"nearest", "linear", "bilinear", "bicubic", "trilinear", "area", "nearest-exact"}

    def __init__(
        self,
        input_path: str,
        size: Optional[Union[int, Tuple[int, int]]] = None,
        take_size_from: Optional[str] = None,
        resize_mode: str = "bilinear",
        patch_mode: bool = False,
        channels_last: bool = False,
    ):
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"tensor": input_path, "size_tensor": take_size_from})

        if size is not None and take_size_from is not None:
            raise ValueError("`size` and `take_size_from` can not be set at the same time")
        self.size = size

        if resize_mode not in Resize.MODES:
            raise ValueError(f"`mode` must be one of {Resize.MODES}")
        self.resize_mode = resize_mode
        self.patch_mode = patch_mode
        self.channels_last = channels_last
        self.expected_dims = 3 if patch_mode else 4

    @RoutableMixin.route
    def forward(
        self, tensor: torch.Tensor, size_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Resize tensor.

        Args:
            tensor: Tensor to resize. If `patch_mode=False`, assumed to be of shape (..., C, H, W).
                If `patch_mode=True`, assumed to be of shape (..., C, P), where P is the number of
                patches. Patches are assumed to be viewable as a perfect square image. If
                `channels_last=True`, channel dimension is assumed to be the last dimension instead.
            size_tensor: Tensor which size to resize to. If tensor has <=2 dimensions and the last
                dimension of this tensor has length 2, the two entries are taken as height and width.
                Otherwise, the size of the last two dimensions of this tensor are used as height
                and width.

        Returns: Tensor of shape (..., C, H, W), where height and width are either specified by
            `size` or `size_tensor`.
        """
        dims_to_flatten = tensor.ndim - self.expected_dims
        if dims_to_flatten > 0:
            flattened_dims = tensor.shape[: dims_to_flatten + 1]
            tensor = tensor.flatten(0, dims_to_flatten)
        elif dims_to_flatten < 0:
            raise ValueError(
                f"Tensor needs at least {self.expected_dims} dimensions, but only has {tensor.ndim}"
            )

        if self.patch_mode:
            if self.channels_last:
                tensor = tensor.transpose(-2, -1)
            n_channels, n_patches = tensor.shape[-2:]
            patch_size_float = math.sqrt(n_patches)
            patch_size = int(math.sqrt(n_patches))
            if patch_size_float != patch_size:
                raise ValueError(
                    f"The number of patches needs to be a perfect square, but is {n_patches}."
                )
            tensor = tensor.view(-1, n_channels, patch_size, patch_size)
        else:
            if self.channels_last:
                tensor = tensor.permute(0, 3, 1, 2)

        if self.size is None:
            if size_tensor is None:
                raise ValueError("`size` is `None` but no `size_tensor` was passed.")
            if size_tensor.ndim <= 2 and size_tensor.shape[-1] == 2:
                height, width = size_tensor.unbind(-1)
                height = torch.atleast_1d(height)[0].squeeze().detach().cpu()
                width = torch.atleast_1d(width)[0].squeeze().detach().cpu()
                size = (int(height), int(width))
            else:
                size = size_tensor.shape[-2:]
        else:
            size = self.size

        tensor = torch.nn.functional.interpolate(
            tensor,
            size=size,
            mode=self.resize_mode,
        )

        if dims_to_flatten > 0:
            tensor = tensor.unflatten(0, flattened_dims)

        return tensor


def resize_patches_to_image(
    patches: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    resize_mode: str = "bilinear",
) -> torch.Tensor:
    """Convert and resize a tensor of patches to image shape.

    This method requires that the patches can be converted to a square image.

    Args:
        patches: Patches to be converted of shape (..., C, P), where C is the number of channels and
            P the number of patches.
        size: Image size to resize to.
        scale_factor: Scale factor by which to resize the patches. Can be specified alternatively to
            `size`.
        resize_mode: Method to resize with. Valid options are "nearest", "nearest-exact", "bilinear",
            "bicubic".

    Returns: Tensor of shape (..., C, S, S) where S is the image size.
    """
    has_size = size is None
    has_scale = scale_factor is None
    if has_size == has_scale:
        raise ValueError("Exactly one of `size` or `scale_factor` must be specified.")

    n_channels = patches.shape[-2]
    n_patches = patches.shape[-1]
    patch_size_float = math.sqrt(n_patches)
    patch_size = int(math.sqrt(n_patches))
    if patch_size_float != patch_size:
        raise ValueError("The number of patches needs to be a perfect square.")

    image = torch.nn.functional.interpolate(
        patches.view(-1, n_channels, patch_size, patch_size),
        size=size,
        scale_factor=scale_factor,
        mode=resize_mode,
    )

    return image.view(*patches.shape[:-1], image.shape[-2], image.shape[-1])
