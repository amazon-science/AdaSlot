"""Configurations for feature extractors."""
import dataclasses

from hydra_zen import make_custom_builds_fn

from ocl import feature_extractors


@dataclasses.dataclass
class FeatureExtractorConfig:
    """Base class for PyTorch Lightning DataModules.

    This class does not actually do anything but ensures that feature extractors give outputs of
    a defined structure.
    """

    pass


builds_feature_extractor = make_custom_builds_fn(
    populate_full_signature=True,
)

TimmFeatureExtractorConfig = builds_feature_extractor(
    feature_extractors.TimmFeatureExtractor,
    builds_bases=(FeatureExtractorConfig,),
)
SlotAttentionFeatureExtractorConfig = builds_feature_extractor(
    feature_extractors.SlotAttentionFeatureExtractor,
    builds_bases=(FeatureExtractorConfig,),
)
DVAEFeatureExtractorConfig = builds_feature_extractor(
    feature_extractors.DVAEFeatureExtractor,
    builds_bases=(FeatureExtractorConfig,),
)
SAViFeatureExtractorConfig = builds_feature_extractor(
    feature_extractors.SAViFeatureExtractor,
    builds_bases=(FeatureExtractorConfig,),
)


def register_configs(config_store):
    config_store.store(group="schemas", name="feature_extractor", node=FeatureExtractorConfig)
    config_store.store(
        group="feature_extractor",
        name="timm_model",
        node=TimmFeatureExtractorConfig,
    )
    config_store.store(
        group="feature_extractor",
        name="slot_attention",
        node=SlotAttentionFeatureExtractorConfig,
    )
    config_store.store(
        group="feature_extractor",
        name="savi",
        node=SAViFeatureExtractorConfig,
    )
    config_store.store(
        group="feature_extractor",
        name="dvae",
        node=DVAEFeatureExtractorConfig,
    )
