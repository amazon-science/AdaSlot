from typing import Any, Dict

from torch import nn

from ocl.path_defaults import VIDEO
from ocl.utils.trees import get_tree_element
import torch

class GroupingImgGumbel(nn.Module):
    def __init__(
        self,
        conditioning: nn.Module,
        feature_extractor: nn.Module,
        perceptual_grouping: nn.Module,
        object_decoder: nn.Module,
        masks_as_image = None,
    ):
        super().__init__()
        self.conditioning = conditioning
        self.feature_extractor = feature_extractor
        self.perceptual_grouping = perceptual_grouping
        self.object_decoder = object_decoder
        self.masks_as_image = masks_as_image
        object_dim = self.conditioning.object_dim
        
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
        outputs["hard_keep_decision"] = perceptual_grouping_output["hard_keep_decision"]
        outputs["slots_keep_prob"] = perceptual_grouping_output["slots_keep_prob"]

        ##
        object_features, hard_keep_decision = perceptual_grouping_output["objects"], perceptual_grouping_output["hard_keep_decision"]         # (b * t, s, d), (b * t, s, n)
        # slot decoding
        # object_features = get_tree_element(outputs, "perceptual_grouping.objects".split("."))
        decoder_output = self.object_decoder(object_features=object_features, 
                                                    left_mask = hard_keep_decision)
   
        outputs["object_decoder"] = decoder_output
        if not self.masks_as_image is None:
            outputs["masks_as_image"]= self.masks_as_image(tensor = get_tree_element(outputs, "object_decoder.masks".split(".")))
        return outputs