"""Implementations of perceptual grouping algorithms."""
import math
from typing import Any, Dict, Optional

import numpy
import torch
from sklearn import cluster
from torch import nn
import torch
from ocl import base, path_defaults
from ocl.utils.routing import RoutableMixin


class SlotAttention(nn.Module):
    """Implementation of SlotAttention.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)

        return slots, attn


class SlotAttentionGrouping(base.PerceptualGrouping, RoutableMixin):
    """Implementation of SlotAttention for perceptual grouping.

    Args:
        feature_dim: Dimensionality of features to slot attention (after positional encoding).
        object_dim: Dimensionality of slots.
        kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
            `object_dim` is used.
        n_heads: Number of heads slot attention uses.
        iters: Number of slot attention iterations.
        eps: Epsilon in slot attention.
        ff_mlp: Optional module applied slot-wise after GRU update.
        positional_embedding: Optional module applied to the features before slot attention, adding
            positional encoding.
        use_projection_bias: Whether to use biases in key, value, query projections.
        use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
            performs one more iteration of slot attention that is used for the gradient step after
            `iters` iterations of slot attention without gradients. Faster and more memory efficient
            than the standard version, but can not backpropagate gradients to the conditioning input.
        input_dim: Dimensionality of features before positional encoding is applied. Specifying this
            is optional but can be convenient to structure configurations.
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        input_dim: Optional[int] = None,
        feature_path: Optional[str] = path_defaults.FEATURES,
        conditioning_path: Optional[str] = path_defaults.CONDITIONING,
        slot_mask_path: Optional[str] = None,
    ):
        base.PerceptualGrouping.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "extracted_features": feature_path,
                "conditioning": conditioning_path,
                "slot_masks": slot_mask_path,
            },
        )

        self._object_dim = object_dim
        self.slot_attention = SlotAttention(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            if slot_mask_path is None:
                raise ValueError("Need `slot_mask_path` for `use_empty_slot_for_masked_slots`")
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None

    @property
    def object_dim(self):
        return self._object_dim

    @RoutableMixin.route
    def forward(
        self,
        extracted_features: base.FeatureExtractorOutput,
        conditioning: base.ConditioningOutput,
        slot_masks: Optional[torch.Tensor] = None,
    ):
        if self.positional_embedding:
            features = self.positional_embedding(
                extracted_features.features, extracted_features.positions
            )
        else:
            features = extracted_features.features

        slots, attn = self.slot_attention(features, conditioning, slot_masks)

        if slot_masks is not None and self.empty_slot is not None:
            slots[slot_masks] = self.empty_slot.to(dtype=slots.dtype)

        return base.PerceptualGroupingOutput(slots, feature_attributions=attn, is_empty=slot_masks)

import torch.nn.functional as F
def sample_slot_lower_bound(A, lower_bound = 1):
    """
    A: [b, k] a batch of slot mask
    0 mean drop, 1 means left
    To make sure at least some slot is choosen
    """
    # A = A.detach()
    B = torch.zeros_like(A, device = A.device)
    batch_slot_leftnum = (A != 0).sum(-1)
    lesser_column_idx = torch.nonzero(batch_slot_leftnum < lower_bound).reshape(-1)
    for j in lesser_column_idx:
        left_slot_mask = A[j]
        sample_slot_zero_idx = torch.nonzero(left_slot_mask==0).reshape(-1)
        # Generate a random permutation of indices
        sampled_indices = torch.randperm(sample_slot_zero_idx.size(0))[:lower_bound - batch_slot_leftnum[j]]
        sampled_elements = sample_slot_zero_idx[sampled_indices]
        B[j][sampled_elements] += 1
    return B

class SlotAttentionGumbelV1(nn.Module):
    """Implementation of SlotAttention with Gumbel Selection Module.

    Based on the slot attention implementation of Phil Wang available at:
    https://github.com/lucidrains/slot-attention
    """

    def __init__(
        self,
        dim: int,
        feature_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        single_gumbel_score_network = None,
        low_bound = 0,
        temporature_function = None
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.iters = iters
        self.eps = eps
        self.use_implicit_differentiation = use_implicit_differentiation

        if kvq_dim is None:
            self.kvq_dim = dim
        else:
            self.kvq_dim = kvq_dim

        if self.kvq_dim % self.n_heads != 0:
            raise ValueError("Key, value, query dimensions must be divisible by number of heads.")
        self.dims_per_head = self.kvq_dim // self.n_heads
        self.scale = self.dims_per_head**-0.5

        self.to_q = nn.Linear(dim, self.kvq_dim, bias=use_projection_bias)
        self.to_k = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)
        self.to_v = nn.Linear(feature_dim, self.kvq_dim, bias=use_projection_bias)

        self.gru = nn.GRUCell(self.kvq_dim, dim)

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.ff_mlp = ff_mlp
        self.single_gumbel_score_network = single_gumbel_score_network
        self.low_bound = low_bound
        self.temporature_function = temporature_function

    def step(self, slots, k, v, masks=None):
        bs, n_slots, _ = slots.shape
        slots_prev = slots

        slots = self.norm_slots(slots)
        q = self.to_q(slots).view(bs, n_slots, self.n_heads, self.dims_per_head)

        dots = torch.einsum("bihd,bjhd->bihj", q, k) * self.scale
        if masks is not None:
            # Masked slots should not take part in the competition for features. By replacing their
            # dot-products with -inf, their attention values will become zero within the softmax.
            dots.masked_fill_(masks.to(torch.bool).view(bs, n_slots, 1, 1), float("-inf"))

        attn = dots.flatten(1, 2).softmax(dim=1)  # Take softmax over slots and heads
        attn = attn.view(bs, n_slots, self.n_heads, -1)
        attn_before_reweighting = attn
        attn = attn + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)

        updates = torch.einsum("bjhd,bihj->bihd", v, attn)

        slots = self.gru(updates.reshape(-1, self.kvq_dim), slots_prev.reshape(-1, self.dim))

        slots = slots.reshape(bs, -1, self.dim)

        if self.ff_mlp:
            slots = self.ff_mlp(slots)

        return slots, attn_before_reweighting.mean(dim=2)

    def iterate(self, slots, k, v, masks=None):
        for _ in range(self.iters):
            slots, attn = self.step(slots, k, v, masks)
        return slots, attn

    def forward(
        self, inputs: torch.Tensor, conditioning: torch.Tensor, masks: Optional[torch.Tensor] = None,
        global_step = None
    ):
        b, n, d = inputs.shape
        slots = conditioning

        inputs = self.norm_input(inputs)
        k = self.to_k(inputs).view(b, n, self.n_heads, self.dims_per_head)
        v = self.to_v(inputs).view(b, n, self.n_heads, self.dims_per_head)

        if self.use_implicit_differentiation:
            slots, attn = self.iterate(slots, k, v, masks)
            slots, attn = self.step(slots.detach(), k, v, masks)
        else:
            slots, attn = self.iterate(slots, k, v, masks)
        
        """
        Gumbel selection
        """
        # b, k, d = conditioning.shape
        _, k, _ = conditioning.shape
        prev_decision = torch.ones(b, k, dtype=slots.dtype, device=slots.device) #prev_decision [b, k]
        
        slots_keep_prob = self.single_gumbel_score_network(slots) #slots_keep_prob [b, k, 2]
        if global_step == None:
            tau = 1
        else:
            tau = self.temporature_function(global_step)
        current_keep_decision = F.gumbel_softmax(slots_keep_prob, hard=True, tau = tau)[...,1]
        if  self.low_bound > 0:
            current_keep_decision = current_keep_decision + sample_slot_lower_bound(current_keep_decision, self.low_bound)
        hard_keep_decision = current_keep_decision * prev_decision #hard_keep_decision [b, k]
        slots_keep_prob = F.softmax(slots_keep_prob, dim = -1)[...,1]
        # hard_idx[...,0]

        return slots, attn, slots_keep_prob, hard_keep_decision


class SlotAttentionGroupingGumbelV1(base.PerceptualGrouping, RoutableMixin):
    """Implementation of SlotAttention for perceptual grouping.

    Args:
        feature_dim: Dimensionality of features to slot attention (after positional encoding).
        object_dim: Dimensionality of slots.
        kvq_dim: Dimensionality after projecting to keys, values, and queries. If `None`,
            `object_dim` is used.
        n_heads: Number of heads slot attention uses.
        iters: Number of slot attention iterations.
        eps: Epsilon in slot attention.
        ff_mlp: Optional module applied slot-wise after GRU update.
        positional_embedding: Optional module applied to the features before slot attention, adding
            positional encoding.
        use_projection_bias: Whether to use biases in key, value, query projections.
        use_implicit_differentiation: Whether to use implicit differentiation trick. If true,
            performs one more iteration of slot attention that is used for the gradient step after
            `iters` iterations of slot attention without gradients. Faster and more memory efficient
            than the standard version, but can not backpropagate gradients to the conditioning input.
        input_dim: Dimensionality of features before positional encoding is applied. Specifying this
            is optional but can be convenient to structure configurations.
    """

    def __init__(
        self,
        feature_dim: int,
        object_dim: int,
        kvq_dim: Optional[int] = None,
        n_heads: int = 1,
        iters: int = 3,
        eps: float = 1e-8,
        ff_mlp: Optional[nn.Module] = None,
        positional_embedding: Optional[nn.Module] = None,
        use_projection_bias: bool = False,
        use_implicit_differentiation: bool = False,
        use_empty_slot_for_masked_slots: bool = False,
        input_dim: Optional[int] = None,
        feature_path: Optional[str] = path_defaults.FEATURES,
        conditioning_path: Optional[str] = path_defaults.CONDITIONING,
        slot_mask_path: Optional[str] = None,
        single_gumbel_score_network: Optional[nn.Module] = None,
        low_bound = 0,
        temporature_function = None
    ):
        base.PerceptualGrouping.__init__(self)
        RoutableMixin.__init__(
            self,
            {
                "extracted_features": feature_path,
                "conditioning": conditioning_path,
                "slot_masks": slot_mask_path,
                "global_step": path_defaults.GLOBAL_STEP
            },
        )

        self._object_dim = object_dim
        self.slot_attention = SlotAttentionGumbelV1(
            dim=object_dim,
            feature_dim=feature_dim,
            kvq_dim=kvq_dim,
            n_heads=n_heads,
            iters=iters,
            eps=eps,
            ff_mlp=ff_mlp,
            use_projection_bias=use_projection_bias,
            use_implicit_differentiation=use_implicit_differentiation,
            single_gumbel_score_network = single_gumbel_score_network,
            low_bound = low_bound,
            temporature_function = temporature_function
        )

        self.positional_embedding = positional_embedding

        if use_empty_slot_for_masked_slots:
            if slot_mask_path is None:
                raise ValueError("Need `slot_mask_path` for `use_empty_slot_for_masked_slots`")
            self.empty_slot = nn.Parameter(torch.randn(object_dim) * object_dim**-0.5)
        else:
            self.empty_slot = None
            
        if temporature_function==None:
            temporature_function = (lambda step: 1)

        self.temporature_function = temporature_function

    @property
    def object_dim(self):
        return self._object_dim

    @RoutableMixin.route
    def forward(
        self,
        extracted_features: base.FeatureExtractorOutput,
        conditioning: base.ConditioningOutput,
        slot_masks: Optional[torch.Tensor] = None,
        global_step = None
    ):
        if self.positional_embedding:
            features = self.positional_embedding(
                extracted_features.features, extracted_features.positions
            )
        else:
            features = extracted_features.features

        slots, attn, slots_keep_prob, hard_keep_decision = self.slot_attention(features, conditioning, slot_masks, global_step = global_step)

        if slot_masks is not None and self.empty_slot is not None:
            slots[slot_masks] = self.empty_slot.to(dtype=slots.dtype)
        
        # objects: TensorType["batch_size", "n_objects", "object_dim"]  # noqa: F821
        # is_empty: Optional[TensorType["batch_size", "n_objects"]] = None  # noqa: F821
        # feature_attributions: Optional[
        #     TensorType["batch_size", "n_objects", "n_spatial_features"]  # noqa: F821
        # ] = None

        return {
            "objects": slots,
            "is_empty": slot_masks,
            "feature_attributions":attn,
            "slots_keep_prob": slots_keep_prob,
            "hard_keep_decision":hard_keep_decision
        }