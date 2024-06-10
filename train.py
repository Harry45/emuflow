import os
import torch
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

# our scripts
from src.flow import build_flow
from src.utils import get_logger
from cfglib import FlowConfig

cs = ConfigStore.instance()
cs.store(name="FlowConfig", node=FlowConfig)


@hydra.main(
    version_base=None, config_path="conf/experiment", config_name="oxford_des_y1"
)
def NFTraining(cfg: FlowConfig):

    # initiate the logger
    logger = get_logger()
    logger.info("-" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-" * 50)

    # train the normalising flow
    flow = build_flow(cfg)
    return flow


if __name__ == "__main__":
    flow = NFTraining()
