"""Implementation of feature extractors."""
import enum
import itertools
import math
from functools import partial
from typing import Callable, List, Optional, Union

import torch
from torch import nn

from ocl import base, path_defaults
from ocl.utils.routing import RoutableMixin


def cnn_compute_positions_and_flatten(features: torch.Tensor):
    """Flatten output image CNN output and return it with positions of the features."""
    # todo(hornmax): see how this works with vision transformer based architectures.
    spatial_dims = features.shape[2:]
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )
    # reorder into format (batch_size, flattened_spatial_dims, feature_dim).
    flattened = torch.permute(features.view(features.shape[:2] + (-1,)), (0, 2, 1)).contiguous()
    return positions, flattened


def transformer_compute_positions(features: torch.Tensor):
    """Compute positions for Transformer features."""
    n_tokens = features.shape[1]
    image_size = math.sqrt(n_tokens)
    image_size_int = int(image_size)
    assert (
        image_size_int == image_size
    ), "Position computation for Transformers requires square image"

    spatial_dims = (image_size_int, image_size_int)
    positions = torch.cartesian_prod(
        *[torch.linspace(0.0, 1.0, steps=dim, device=features.device) for dim in spatial_dims]
    )

    return positions


class ImageFeatureExtractor(base.FeatureExtractor, RoutableMixin):
    """Feature extractor which operates on images.

    For these we reshape the frame dimension into the batch dimension and process the frames as
    individual images.
    """

    def __init__(self, video_path: Optional[str] = path_defaults.VIDEO):
        base.FeatureExtractor.__init__(self)
        RoutableMixin.__init__(self, {"video": video_path, "global_step": path_defaults.GLOBAL_STEP})

    def forward_images(self, images: torch.Tensor):
        pass

    @RoutableMixin.route
    def forward(self, video: torch.Tensor) -> base.FeatureExtractorOutput:
        # print("video.shape")
        # print(video.shape)

        ndim = video.dim()
        assert ndim == 4 or ndim == 5

        if ndim == 5:
            # Handling video data.
            bs, frames, channels, height, width = video.shape
            images = video.view(bs * frames, channels, height, width).contiguous()
        else:
            images = video

        result = self.forward_images(images)

        if len(result) == 2:
            positions, features = result
            aux_features = None
        elif len(result) == 3:
            positions, features, aux_features = result

        if ndim == 5:
            features = features.unflatten(0, (bs, frames))
            if aux_features is not None:
                aux_features = {k: f.unflatten(0, (bs, frames)) for k, f in aux_features.items()}

        return base.FeatureExtractorOutput(features, positions, aux_features)


class ClipImageModel(nn.Module, RoutableMixin):
    def __init__(
        self,
        model_type: str,
        image_path: Optional[str] = path_defaults.VIDEO,
        freeze_model: bool = False,
        reset_weights: bool = False,
        remove_pooling: bool = False,
    ):
        try:
            import clip
        except ImportError:
            raise Exception("Using clip models requires installation with extra `clip`.")
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"image": image_path})
        self.freeze_model = freeze_model

        self.clip_vision_model = clip.load(
            model_type,
            # Initially force cpu to ensure tensors are float32 (load routine automatically converts
            # to half precision if GPUs are detected).  We can still do half-precision training via
            # pytorch lightning if we want to.
            device="cpu",
        )[0].visual
        if self.freeze_model:
            for parameter in self.clip_vision_model.parameters():
                parameter.requires_grad_(False)

        if reset_weights:

            def weight_reset(module):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            self.clip_vision_model.apply(weight_reset)
            self.clip_vision_model.initialize_parameters()

        if remove_pooling:
            if isinstance(self.clip_vision_model, clip.model.VisionTransformer):
                self.get_output = self._get_features_from_vision_transformer
            else:
                self.get_output = self._get_features_from_resnet
        else:
            self.get_output = self.clip_vision_model

    def _get_features_from_vision_transformer(self, x):
        # Commands from:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L223
        model = self.clip_vision_model

        x = model.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [
                model.class_embedding
                + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + model.positional_embedding
        x = model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        return x

    def _get_features_from_resnet(self, x):
        # Commands from:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L138

        model = self.clip_vision_model
        # Apply "stem".
        x = model.relu1(model.bn1(model.conv1(x)))
        x = model.relu2(model.bn2(model.conv2(x)))
        x = model.relu3(model.bn3(model.conv3(x)))
        x = model.avgpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        return x

    @RoutableMixin.route
    def forward(self, image: torch.Tensor):
        if self.freeze_model:
            with torch.no_grad():
                return self.get_output(image)
        else:
            return self.get_output(image)


class ClipTextModel(nn.Module, RoutableMixin):
    def __init__(
        self,
        model_type: str,
        text_path: Optional[str] = path_defaults.TEXT,
        freeze_model: bool = False,
        reset_weights: bool = False,
        remove_pooling: bool = False,
        remove_eot: bool = False,
    ):
        try:
            import clip
        except ImportError:
            raise Exception("Using clip models requires installation with extra `clip`.")
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"text": text_path})
        self.freeze_model = freeze_model
        self.remove_pooling = remove_pooling

        clip_model = clip.load(
            model_type,
            # Initially force cpu to ensure tensors are float32 (load routine automatically converts
            # to half precision if GPUs are detected).  We can still do half-precision training via
            # pytorch lightning if we want to.
            device="cpu",
        )[0]
        if reset_weights:

            def weight_reset(module):
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            clip_model.apply(weight_reset)
            clip_model.initialize_parameters()

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        if self.freeze_model:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

        self.remove_pooling = remove_pooling
        self.remove_eot = remove_eot

    def get_output(self, text):
        # Based on:
        # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L343
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.remove_pooling:
            # Mask out tokens which are part of the padding.
            # Get position of eot token, it has the highest value of all tokens.
            lengths = text.argmax(dim=-1)
            if self.remove_eot:
                # Also mask out the eot token.
                lengths = lengths - 1
            indices = torch.arange(x.shape[1], device=text.device)
            mask = indices.unsqueeze(0) >= lengths
            x.masked_fill_(mask, 0.0)

            x = x @ self.text_projection
        else:
            # Do what is done in the standard clip text encoder.
            # x.shape = [batch_size, n_ctx, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    @RoutableMixin.route
    def forward(self, text: torch.Tensor):
        if self.freeze_model:
            with torch.no_grad():
                return self.get_output(text)
        else:
            return self.get_output(text)


class ClipFeatureExtractor(nn.Module, RoutableMixin):
    def __init__(
        self,
        model_type: str,
        image_path: Optional[str] = path_defaults.VIDEO,
        text_path: Optional[str] = path_defaults.TEXT,
        keep_image_model: bool = True,
        keep_text_model: bool = True,
    ):
        try:
            import clip
        except ImportError:
            raise Exception("Using clip models requires installation with extra `clip`.")
        nn.Module.__init__(self)
        RoutableMixin.__init__(self, {"image": image_path, "text": text_path})
        # Load returns model and preprocessing function, we only want the former.
        clip_model = clip.load(
            model_type,
            # Initially force cpu to ensure tensors are float32 (load routine automatically converts
            # to half precision if GPUs are detected).  We can still do half-precision training via
            # pytorch lightning if we want to.
            device="cpu",
        )[0]
        # TODO: Continue here
        clip_model.do_something()

    @RoutableMixin.route
    def forward(self, image: torch.Tensor, text: torch.Tensor):
        return {
            "image": self.clip_model.encode_image(image),
            "text": self.clip_model.encode_text(text),
        }


class VitFeatureType(enum.Enum):
    BLOCK = 1
    KEY = 2
    VALUE = 3
    QUERY = 4
    CLS = 5


class VitFeatureHook:
    """Auxilliary class used to extract features from timm ViT models.

    Args:
        mode: Type of feature to extract.
        block: Number of block to extract features from. Note that this is not zero-indexed.
    """

    def __init__(self, feature_type: VitFeatureType, block: int, drop_cls_token: bool = True):
        assert isinstance(feature_type, VitFeatureType)
        self.feature_type = feature_type
        self.block = block
        self.drop_cls_token = drop_cls_token
        self.name = f"{feature_type.name.lower()}{block}"
        self.remove_handle = None  # Can be used to remove this hook from the model again

        self._features = None

    @staticmethod
    def create_hook_from_feature_level(feature_level: Union[int, str]):
        feature_level = str(feature_level)
        prefixes = ("key", "query", "value", "block", "cls")
        for prefix in prefixes:
            if feature_level.startswith(prefix):
                _, _, block = feature_level.partition(prefix)
                feature_type = VitFeatureType[prefix.upper()]
                block = int(block)
                break
        else:
            feature_type = VitFeatureType.BLOCK
            try:
                block = int(feature_level)
            except ValueError:
                raise ValueError(f"Can not interpret feature_level '{feature_level}'.")

        return VitFeatureHook(feature_type, block)

    def register_with(self, model):
        import timm

        supported_models = (
            timm.models.vision_transformer.VisionTransformer,
            timm.models.beit.Beit,
        )
        if not isinstance(model, supported_models):
            raise ValueError(
                f"This hook only supports classes {', '.join(str(cl) for cl in supported_models)}."
            )

        if self.block > len(model.blocks):
            raise ValueError(
                f"Trying to extract features of block {self.block}, but model only has "
                f"{len(model.blocks)} blocks"
            )

        block = model.blocks[self.block - 1]
        if self.feature_type == VitFeatureType.BLOCK:
            self.remove_handle = block.register_forward_hook(self)
        else:
            if isinstance(block, timm.models.vision_transformer.ParallelBlock):
                raise ValueError(
                    f"ViT with `ParallelBlock` not supported for {self.feature_type} extraction."
                )
            elif isinstance(model, timm.models.beit.Beit):
                raise ValueError(f"BEIT not supported for {self.feature_type} extraction.")
            self.remove_handle = block.attn.qkv.register_forward_hook(self)

        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook.

        We only allow access to the features this way to not have any lingering references to them.
        """
        assert self._features is not None, "Feature extractor was not called yet!"
        features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp):
        if self.feature_type == VitFeatureType.BLOCK:
            features = outp
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type in {VitFeatureType.KEY, VitFeatureType.QUERY, VitFeatureType.VALUE}:
            # This part is adapted from the timm implementation. Unfortunately, there is no more
            # elegant way to access keys, values, or queries.
            B, N, C = inp[0].shape
            qkv = outp.reshape(B, N, 3, C)  # outp has shape B, N, 3 * H * (C // H)
            q, k, v = qkv.unbind(2)

            if self.feature_type == VitFeatureType.QUERY:
                features = q
            elif self.feature_type == VitFeatureType.KEY:
                features = k
            else:
                features = v
            if self.drop_cls_token:
                # First token is CLS token.
                features = features[:, 1:]
        elif self.feature_type == VitFeatureType.CLS:
            # We ignore self.drop_cls_token in this case as it doesn't make any sense.
            features = outp[:, 0]  # Only get class token.
        else:
            raise ValueError("Invalid VitFeatureType provided.")

        self._features = features


class TimmFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor implementation for timm models.

    Args:
        model_name: Name of model. See `timm.list_models("*")` for available options.
        feature_level: Level of features to return. For CNN-based models, a single integer. For ViT
            models, either a single or a list of feature descriptors. If a list is passed, multiple
            levels of features are extracted and concatenated. A ViT feature descriptor consists of
            the type of feature to extract, followed by an integer indicating the ViT block whose
            features to use. The type of features can be one of "block", "key", "query", "value",
            specifying that the block's output, attention keys, query or value should be used. If
            omitted, assumes "block" as the type. Example: "block1" or ["block1", "value2"].
        aux_features: Features to store as auxilliary features. The format is the same as in the
            `feature_level` argument. Features are stored as a dictionary, using their string
            representation (e.g. "block1") as the key. Only valid for ViT models.
        pretrained: Whether to load pretrained weights.
        freeze: Whether the weights of the feature extractor should be trainable.
        n_blocks_to_unfreeze: Number of blocks that should be trainable, beginning from the last
            block.
        unfreeze_attention: Whether weights of ViT attention layers should be trainable (only valid
            for ViT models). According to http://arxiv.org/abs/2203.09795, finetuning attention
            layers only can yield better results in some cases, while being slightly cheaper in terms
            of computation and memory.
    """

    def __init__(
        self,
        model_name: str,
        feature_level: Optional[Union[int, str, List[Union[int, str]]]] = None,
        aux_features: Optional[Union[int, str, List[Union[int, str]]]] = None,
        pretrained: bool = False,
        freeze: bool = False,
        n_blocks_to_unfreeze: int = 0,
        unfreeze_attention: bool = False,
        video_path: Optional[str] = path_defaults.VIDEO,
    ):
        super().__init__(video_path)
        try:
            import timm
        except ImportError:
            raise Exception("Using timm models requires installation with extra `timm`.")

        register_custom_timm_models()

        self.is_vit = model_name.startswith("vit") or model_name.startswith("beit")

        def feature_level_to_list(feature_level):
            if feature_level is None:
                return []
            elif isinstance(feature_level, (int, str)):
                return [feature_level]
            else:
                return list(feature_level)

        self.feature_levels = feature_level_to_list(feature_level)
        self.aux_features = feature_level_to_list(aux_features)

        if self.is_vit:
            model = timm.create_model(model_name, pretrained=pretrained)
            # Delete unused parameters from classification head
            if hasattr(model, "head"):
                del model.head
            if hasattr(model, "fc_norm"):
                del model.fc_norm

            if len(self.feature_levels) > 0 or len(self.aux_features) > 0:
                self._feature_hooks = [
                    VitFeatureHook.create_hook_from_feature_level(level).register_with(model)
                    for level in itertools.chain(self.feature_levels, self.aux_features)
                ]
                if len(self.feature_levels) > 0:
                    feature_dim = model.num_features * len(self.feature_levels)

                    # Remove modules not needed in computation of features
                    max_block = max(hook.block for hook in self._feature_hooks)
                    new_blocks = model.blocks[:max_block]  # Creates a copy
                    del model.blocks
                    model.blocks = new_blocks
                    model.norm = nn.Identity()
                else:
                    feature_dim = model.num_features
            else:
                self._feature_hooks = None
                feature_dim = model.num_features
        else:
            if len(self.feature_levels) == 0:
                raise ValueError(
                    f"Feature extractor {model_name} requires specifying `feature_level`"
                )
            elif len(self.feature_levels) != 1:
                raise ValueError(
                    f"Feature extractor {model_name} only supports a single `feature_level`"
                )
            elif not isinstance(self.feature_levels[0], int):
                raise ValueError("`feature_level` needs to be an integer")

            if len(self.aux_features) > 0:
                raise ValueError("`aux_features` not supported by feature extractor {model_name}")

            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=self.feature_levels,
            )
            feature_dim = model.feature_info.channels()[0]

        self.model = model
        self.freeze = freeze
        self.n_blocks_to_unfreeze = n_blocks_to_unfreeze
        self._feature_dim = feature_dim

        if freeze:
            self.model.requires_grad_(False)
            # BatchNorm layers update their statistics in train mode. This is probably not desired
            # when the model is supposed to be frozen.
            contains_bn = any(
                isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
                for m in self.model.modules()
            )
            self.run_in_eval_mode = contains_bn
        else:
            self.run_in_eval_mode = False

        if self.n_blocks_to_unfreeze > 0:
            if not self.is_vit:
                raise NotImplementedError(
                    "`unfreeze_n_blocks` option only implemented for ViT models"
                )
            self.model.blocks[-self.n_blocks_to_unfreeze :].requires_grad_(True)
            if self.model.norm is not None:
                self.model.norm.requires_grad_(True)

        if unfreeze_attention:
            if not self.is_vit:
                raise ValueError("`unfreeze_attention` option only works with ViT models")
            for module in self.model.modules():
                if isinstance(module, timm.models.vision_transformer.Attention):
                    module.requires_grad_(True)

    @property
    def feature_dim(self):
        return self._feature_dim

    def forward_images(self, images: torch.Tensor):
        if self.run_in_eval_mode and self.training:
            self.eval()

        if self.is_vit:
            if self.freeze and self.n_blocks_to_unfreeze == 0:
                # Speed things up a bit by not requiring grad computation.
                with torch.no_grad():
                    features = self.model.forward_features(images)
            else:
                features = self.model.forward_features(images)

            if self._feature_hooks is not None:
                hook_features = [hook.pop() for hook in self._feature_hooks]

            if len(self.feature_levels) == 0:
                # Remove class token when not using hooks.
                features = features[:, 1:]
                positions = transformer_compute_positions(features)
            else:
                features = hook_features[: len(self.feature_levels)]
                positions = transformer_compute_positions(features[0])
                features = torch.cat(features, dim=-1)

            if len(self.aux_features) > 0:
                aux_hooks = self._feature_hooks[len(self.feature_levels) :]
                aux_features = hook_features[len(self.feature_levels) :]
                aux_features = {hook.name: feat for hook, feat in zip(aux_hooks, aux_features)}
            else:
                aux_features = None
        else:
            features = self.model(images)[0]
            positions, features = cnn_compute_positions_and_flatten(features)
            aux_features = None

        return positions, features, aux_features


class SlotAttentionFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor as used in slot attention paper."""

    def __init__(self, video_path: Optional[str] = path_defaults.VIDEO):
        super().__init__(video_path)
        self.layers = nn.Sequential(
            nn.Conv2d(3, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            nn.ReLU(inplace=True),
        )

    @property
    def feature_dim(self):
        return 64

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(features)
        return positions, flattened


class SAViFeatureExtractor(ImageFeatureExtractor):
    """Feature extractor as used in the slot attention for video paper."""

    def __init__(self, larger_input_arch=False, video_path: Optional[str] = path_defaults.VIDEO):
        """Feature extractor as used in the slot attention for video paper.

        Args:
            larger_input_arch: Use the architecture for larger image datasets such as MOVi++, which
                contains more a stride in the first layer and a higher number of feature channels in
                the CNN backbone.
            video_path: Path of input video or also image.
        """
        super().__init__(video_path=video_path)
        self.larger_input_arch = larger_input_arch
        if larger_input_arch:
            self.layers = nn.Sequential(
                # Pytorch does not support stride>1 with padding=same.
                # Implement tensorflow behaviour manually.
                # See: https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
                nn.ZeroPad2d((1, 2, 1, 2)),
                nn.Conv2d(3, out_channels=64, kernel_size=5, stride=2, padding="valid"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_channels=64, kernel_size=5, padding="same"),
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(3, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, out_channels=32, kernel_size=5, padding="same"),
            )

    @property
    def feature_dim(self):
        return 64 if self.larger_input_arch else 32

    def forward_images(self, images: torch.Tensor):
        features = self.layers(images)
        positions, flattened = cnn_compute_positions_and_flatten(features)
        return positions, flattened


def register_custom_timm_models():
    import timm
    from timm.models import layers, resnet, vision_transformer
    from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg

    @timm.models.registry.register_model
    def resnet34_savi(pretrained=False, **kwargs):
        """ResNet34 as used in SAVi and SAVi++.

        As of now, no official code including the ResNet was released, so we can only guess which of
        the numerous ResNet variants was used. This modifies the basic timm ResNet34 to have 1x1
        strides in the stem, and replaces batch norm with group norm. Gives 16x16 feature maps with
        an input size of 224x224.

        From SAVi:
        > For the modified SAVi (ResNet) model on MOVi++, we replace the convolutional backbone [...]
        > with a ResNet-34 backbone. We use a modified ResNet root block without strides
        > (i.e. 1×1 stride), resulting in 16×16 feature maps after the backbone [w. 128x128 images].
        > We further use group normalization throughout the ResNet backbone.

        From SAVi++:
        > We used a ResNet-34 backbone with modified root convolutional layer that has 1×1 stride.
        > For all layers, we replaced the batch normalization operation by group normalization.
        """
        if pretrained:
            raise ValueError("No pretrained weights available for `savi_resnet34`.")

        model_args = dict(
            block=resnet.BasicBlock, layers=[3, 4, 6, 3], norm_layer=layers.GroupNorm, **kwargs
        )
        model = resnet._create_resnet("resnet34", pretrained=pretrained, **model_args)
        model.conv1.stride = (1, 1)
        model.maxpool.stride = (1, 1)
        return model

    @timm.models.registry.register_model
    def resnet50_dino(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = resnet._cfg(
            url=(
                "https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/"
                "dino_resnet50_pretrain.pth"
            )
        )
        model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        return build_model_with_cfg(resnet.ResNet, "resnet50_dino", pretrained, **model_args)

    def add_moco_positional_embedding(model, temperature=10000.0):
        """Moco ViT uses 2d sincos embedding."""
        h, w = model.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            model.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = model.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1
        )[None, :, :]
        if hasattr(model, "num_tokens"):  # Old timm versions
            assert model.num_tokens == 1, "Assuming one and only one token, [cls]"
        else:
            assert model.num_prefix_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
        model.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        model.pos_embed.requires_grad = False

    def moco_checkpoint_filter_fn(state_dict, model, linear_name):
        state_dict = state_dict["state_dict"]

        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith("module.base_encoder") and not k.startswith(
                f"module.base_encoder.{linear_name}"
            ):
                # remove prefix
                state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        return state_dict

    def create_moco_vit(variant, pretrained=False, **kwargs):
        if kwargs.get("features_only", None):
            raise RuntimeError("features_only not implemented for Vision Transformer models.")

        pretrained_cfg = resolve_pretrained_cfg(
            variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
        )
        model = build_model_with_cfg(
            vision_transformer.VisionTransformer,
            variant,
            pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_filter_fn=partial(moco_checkpoint_filter_fn, linear_name="head"),
            pretrained_custom_load=False,
            **kwargs,
        )
        add_moco_positional_embedding(model)
        return model

    @timm.models.registry.register_model
    def vit_small_patch16_224_mocov3(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/moco-v3/vit-s-300ep/vit-s-300ep.pth.tar"
        )
        model_kwargs = dict(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
        model = create_moco_vit(
            "vit_small_patch16_224_mocov3", pretrained=pretrained, **model_kwargs
        )
        return model

    @timm.models.registry.register_model
    def vit_base_patch16_224_mocov3(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/vit-b-300ep.pth.tar"
        )
        model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
        model = create_moco_vit("vit_base_patch16_224_mocov3", pretrained=pretrained, **model_kwargs)
        return model

    @timm.models.registry.register_model
    def resnet50_mocov3(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = resnet._cfg(
            url="https://dl.fbaipublicfiles.com/moco-v3/r-50-1000ep/r-50-1000ep.pth.tar"
        )
        model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
        return build_model_with_cfg(
            resnet.ResNet,
            "resnet50_mocov3",
            pretrained,
            pretrained_filter_fn=partial(moco_checkpoint_filter_fn, linear_name="fc"),
            **model_args,
        )

    def msn_vit_checkpoint_filter_fn(state_dict, model):
        state_dict = state_dict["target_encoder"]

        for k in list(state_dict.keys()):
            if not k.startswith("module.fc."):
                # remove prefix
                state_dict[k[len("module.") :]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        return state_dict

    def create_msn_vit(variant, pretrained=False, **kwargs):
        if kwargs.get("features_only", None):
            raise RuntimeError("features_only not implemented for Vision Transformer models.")

        pretrained_cfg = resolve_pretrained_cfg(
            variant, pretrained_cfg=kwargs.pop("pretrained_cfg", None)
        )
        model = build_model_with_cfg(
            vision_transformer.VisionTransformer,
            variant,
            pretrained,
            pretrained_cfg=pretrained_cfg,
            pretrained_filter_fn=msn_vit_checkpoint_filter_fn,
            pretrained_custom_load=False,
            **kwargs,
        )
        return model

    @timm.models.registry.register_model
    def vit_small_patch16_224_msn(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/msn/vits16_800ep.pth.tar"
        )
        model_kwargs = dict(
            patch_size=16,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
        model = create_msn_vit("vit_small_patch16_224_msn", pretrained=pretrained, **model_kwargs)
        return model

    @timm.models.registry.register_model
    def vit_base_patch16_224_msn(pretrained=False, **kwargs):
        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/msn/vitb16_600ep.pth.tar"
        )
        model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
        model = create_msn_vit("vit_base_patch16_224_msn", pretrained=pretrained, **model_kwargs)
        return model

    @timm.models.registry.register_model
    def vit_base_patch16_224_mae(pretrained=False, **kwargs):
        from timm.models.vision_transformer import _create_vision_transformer

        kwargs["pretrained_cfg"] = vision_transformer._cfg(
            url="https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
        )
        model_kwargs = dict(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            **kwargs,
        )
        model = _create_vision_transformer(
            "vit_base_patch16_224_mae", pretrained=pretrained, **model_kwargs
        )
        return model


class DVAEFeatureExtractor(ImageFeatureExtractor):
    """DVAE VQ Encoder in SLATE."""

    def __init__(
        self,
        encoder: nn.Module,
        positional_encoder: nn.Module,
        dictionary: nn.Module,
        tau: Callable,
        hard: bool = False,
        video_path: Optional[str] = path_defaults.VIDEO,
    ):
        """Feature extractor as used in the SLATE paper.

        Args:
            encoder: torch Module that transforms image to the patch representations.
            positional_encoder: torch Module that adds pos encoding.
            dictionary: map from onehot vectors to embeddings.
            tau: temporature for gumbel_softmax.
            hard: hard gumbel_softmax if True.
            video_path: path to original inputs.
        """
        super().__init__(video_path)
        self.global_step = None
        self.tau = tau
        self.hard = hard
        self.dictionary = dictionary
        self.positional_encoder = positional_encoder
        self.encoder = encoder

    @property
    def feature_dim(self):
        return 64

    def forward_images(self, images: torch.Tensor):
        z_logits = nn.functional.log_softmax(self.encoder(images), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = nn.functional.gumbel_softmax(z_logits, self.tau(self.global_step), self.hard, dim=1)
        z_hard = nn.functional.gumbel_softmax(
            z_logits, self.tau(self.global_step), True, dim=1
        ).detach()

        # add beginning of sequence (BOS) token
        # [1, 0, 0, 0, ...] is encoding for BOS token
        # and each sequence starts from such token
        z_hard = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        # add first zeros column to the z_hard matrix
        z_transformer_input = torch.cat([torch.zeros_like(z_hard[..., :1]), z_hard], dim=-1)
        # add first zeros row to the z_hard matrix
        z_transformer_input = torch.cat(
            [torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2
        )
        # fill new row and column with one,
        # so that we added [1, 0, 0, 0, ...] token
        z_transformer_input[:, 0, 0] = 1.0

        # tokens to embeddings
        features = self.dictionary(z_transformer_input)
        features = self.positional_encoder(features)

        slot_attention_features = features[:, 1:]

        transformer_input = features[:, :-1]
        aux_features = {
            "z": z,
            "targets": transformer_input,
            "z_hard": z_hard,
        }
        return None, slot_attention_features, aux_features

    @RoutableMixin.route
    def forward(self, video: torch.Tensor, global_step: int) -> base.FeatureExtractorOutput:
        self.global_step = global_step
        return super().forward(video=video)
