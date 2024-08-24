"""Register all dataset related configs."""
import dataclasses
import os

import yaml
from hydra.utils import to_absolute_path
from hydra_zen import builds

from ocl import datasets


def get_region():
    """Determine the region this EC2 instance is running in.

    Returns None if not running on an EC2 instance.
    """
    import requests

    try:
        r = requests.get(
            "http://169.254.169.254/latest/dynamic/instance-identity/document", timeout=0.5
        )
        response_json = r.json()
        return response_json.get("region")
    except Exception:
        # Not running on an ec2 instance.
        return None


# Detemine region name and select bucket accordingly.
AWS_REGION = get_region()
if AWS_REGION in ["us-east-2", "us-west-2", "eu-west-1"]:
    # Select bucket in same region.
    DEFAULT_S3_PATH = f"s3://object-centric-datasets-{AWS_REGION}"
    # fanke aws s3 ls s3://object-centric-datasets-us-west-2/clevr_with_masks_new_splits
    # aws s3 cp --recursive s3://object-centric-datasets-us-west-2/clevr_with_masks_new_splits ./clevr_with_masks_new_splits
    # # aws s3 ls s3://object-centric-datasets-us-west-2/
    # aws s3 cp --recursive s3://object-centric-datasets-us-west-2/movi_e/ movi_e
else:
    # Use MRAP to find closest bucket.
    DEFAULT_S3_PATH = "s3://arn:aws:s3::436622332146:accesspoint/m6p4hmmybeu97.mrap"


@dataclasses.dataclass
class DataModuleConfig:
    """Base class for PyTorch Lightning DataModules.

    This class does not actually do anything but ensures that datasets behave like pytorch lightning
    datamodules.
    """


def dataset_prefix(path):
    prefix = os.environ.get("DATASET_PREFIX")
    if prefix:
        return f"{prefix}/{path}"
    # Use the path to the multi-region bucket if no override is specified.
    return f"pipe:aws s3 cp --quiet {DEFAULT_S3_PATH}/{path} -"


def read_yaml(path):
    with open(to_absolute_path(path), "r") as f:
        return yaml.safe_load(f)


WebdatasetDataModuleConfig = builds(
    datasets.WebdatasetDataModule, populate_full_signature=True, builds_bases=(DataModuleConfig,)
)
DummyDataModuleConfig = builds(
    datasets.DummyDataModule, populate_full_signature=True, builds_bases=(DataModuleConfig,)
)


def register_configs(config_store):
    config_store.store(group="schemas", name="dataset", node=DataModuleConfig)
    config_store.store(group="dataset", name="webdataset", node=WebdatasetDataModuleConfig)
    config_store.store(group="dataset", name="dummy_dataset", node=DummyDataModuleConfig)


def register_resolvers(omegaconf):
    omegaconf.register_new_resolver("dataset_prefix", dataset_prefix)
    omegaconf.register_new_resolver("read_yaml", read_yaml)
