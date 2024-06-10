import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf
from src.joint import SampleFlow, NormFlow
from src.utils import dill_save, get_logger
from cfglib import SamplerConfig

cs = ConfigStore.instance()
cs.store(name="MCMCconfig", node=SamplerConfig)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def NFSampler(cfg):

    # initiate the logger
    logger = get_logger()
    logger.info("-" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-" * 50)

    # Sample the Joint
    joint_flow = SampleFlow(cfg.joints)
    sampler_flow = joint_flow.sampler(cfg.nmcmc, cfg.eps)
    dill_save(sampler_flow, cfg.mcmc_fname)
    return sampler_flow


if __name__ == "__main__":
    sampler_flow = NFSampler()
