from typing import Any, Dict

from torch import nn

from ocl.path_defaults import VIDEO
from ocl.utils.trees import get_tree_element


class GroupingImg(nn.Module):
    def __init__(
        self,
        conditioning: nn.Module,
        feature_extractor: nn.Module,
        perceptual_grouping: nn.Module,
        object_decoder: nn.Module,
        masks_as_image = None,
        decoder_mode = "MLP",

    ):
        super().__init__()
        self.conditioning = conditioning
        self.feature_extractor = feature_extractor
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.masks_as_image = masks_as_image
        self.decoder_mode = decoder_mode

    def forward(self, inputs: Dict[str, Any]):
        outputs = inputs
        video = get_tree_element(inputs, VIDEO.split("."))
        video.shape

        # feature extraction
        features = self.feature_extractor(video=video)
        outputs["feature_extractor"] = features

        # slot initialization
        batch_size = video.shape[0]
        conditioning = self.conditioning(batch_size=batch_size)
        outputs["conditioning"] = conditioning

        # slot computation
        perceptual_grouping_output = self.perceptual_grouping(
            extracted_features=features, conditioning=conditioning
        )
        outputs["perceptual_grouping"] = perceptual_grouping_output

        # slot decoding
        object_features = get_tree_element(outputs, "perceptual_grouping.objects".split("."))
        masks = get_tree_element(outputs, "perceptual_grouping.feature_attributions".split("."))
        target = get_tree_element(outputs, "feature_extractor.features".split("."))
        image = get_tree_element(outputs, "input.image".split("."))
        empty_object = None

        if self.decoder_mode == "MLP":
            decoder_output = self.object_decoder(object_features=object_features, 
                                                    target=target,
                                                    image = image)
        elif self.decoder_mode == "Transformer":
            decoder_output = self.object_decoder(object_features=object_features,
                                                    masks=masks,
                                                    target=target,
                                                    image=image,
                                                    empty_objects = None)
        else:
            raise RuntimeError
   
        outputs["object_decoder"] = decoder_output
        outputs["masks_as_image"]= self.masks_as_image(tensor = get_tree_element(outputs, "object_decoder.masks".split(".")))

        return outputs
