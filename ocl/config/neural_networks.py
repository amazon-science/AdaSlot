"""Configs for neural networks."""
import omegaconf
from hydra_zen import builds

from ocl import neural_networks

MLPBuilderConfig = builds(
    neural_networks.build_mlp,
    features=omegaconf.MISSING,
    zen_partial=True,
    populate_full_signature=True,
)
TransformerEncoderBuilderConfig = builds(
    neural_networks.build_transformer_encoder,
    n_layers=omegaconf.MISSING,
    n_heads=omegaconf.MISSING,
    zen_partial=True,
    populate_full_signature=True,
)
TransformerDecoderBuilderConfig = builds(
    neural_networks.build_transformer_decoder,
    n_layers=omegaconf.MISSING,
    n_heads=omegaconf.MISSING,
    zen_partial=True,
    populate_full_signature=True,
)


def register_configs(config_store):
    config_store.store(group="neural_networks", name="mlp", node=MLPBuilderConfig)
    config_store.store(
        group="neural_networks", name="transformer_encoder", node=TransformerEncoderBuilderConfig
    )
    config_store.store(
        group="neural_networks", name="transformer_decoder", node=TransformerDecoderBuilderConfig
    )
