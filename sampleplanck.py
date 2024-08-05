import os
import torch
import hydra
from cobaya.run import run
from omegaconf import OmegaConf
from hydra.core.config_store import ConfigStore

# our scripts
from cfglib import DESY1PlanckConfig
from src.desplanck import get_params_info_planck
from planck.model import PlanckLitePy, plite_loglike
from src.utils import get_logger, create_experiment_path

cs = ConfigStore.instance()
cs.store(name="FlowConfig", node=DESY1PlanckConfig)

@hydra.main(version_base=None, config_path="conf", config_name="desy1planck")
def SamplePlanck(cfg: DESY1PlanckConfig):

    # initiate the logger
    logger = get_logger()
    logger.info("-" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-" * 50)

    likelihood = PlanckLitePy(
        data_directory=cfg.planckdata,
        year=cfg.year,
        spectra=cfg.spectra,
        use_low_ell_bins=cfg.use_low_ell_bins,
    )

    def planck_exact_logl(sigma8, omegac, omegab, hubble, ns):
        """
        Computes the exact log-likelihood for Planck data given cosmological parameters.

        This function evaluates the exact log-likelihood of the Planck dataset based on the provided
        cosmological parameters. It utilizes the `plite_loglike` function and a predefined likelihood
        configuration to compute the likelihood.

        Args:
            sigma8 (float): Amplitude of matter fluctuations on 8 Mpc/h scales.
            omegac (float): Cold dark matter density parameter.
            omegab (float): Baryon density parameter.
            hubble (float): Dimensionless Hubble parameter (H0 / 100).
            ns (float): Scalar spectral index.

        Returns:
            float: The log-likelihood value for the given cosmological parameters.
        """
        logl = plite_loglike(likelihood, [sigma8, omegac, omegab, hubble, ns], cfg)
        return logl

    # setup cobaya
    create_experiment_path(cfg)
    info = {"likelihood": {"planck-like": planck_exact_logl}}
    info["params"] = get_params_info_planck(cfg)
    info["output"] = os.path.join(cfg.output_folder, cfg.output_name, cfg.output_name)
    info["sampler"] = {
        "mcmc": {
            "max_samples": cfg.nsamples,
            "Rminus1_stop": 0.01,
        }
    }
    # normal Python run
    updated_info, sampler = run(info)


if __name__ == "__main__":
    samples = SamplePlanck()