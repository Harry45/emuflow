import os
import jax
import hydra
import numpy as np
import jax_cosmo as jc
import jax.numpy as jnp
from cobaya.run import run
from omegaconf import OmegaConf
from jax.lib import xla_bridge
from hydra.core.config_store import ConfigStore

# our script
from cfglib import DESY1PlanckConfig
from src.desplanck import get_params_info_des, load_data
from cosmology.bandpowers import get_bandpowers_theory
from planck.model import PlanckLitePy, plite_loglike
from src.utils import load_flow, get_logger, create_experiment_path

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

print(xla_bridge.get_backend().platform)

@hydra.main(version_base=None, config_path="conf", config_name="desy1planck")
def SampleJoint(cfg: DESY1PlanckConfig):
    # initiate the logger
    logger = get_logger()
    logger.info("-" * 50)
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("-" * 50)

    jc.power.USE_EMU = cfg.use_emu

    DATA, PRECISION, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL = load_data(
    fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000
)
    FLOW = load_flow(cfg.flow_name)

    likelihood = PlanckLitePy(
        data_directory=cfg.planckdata,
        year=cfg.year,
        spectra=cfg.spectra,
        use_low_ell_bins=cfg.use_low_ell_bins,
    )

    @jax.jit
    def jit_theory(parameters: np.ndarray) -> np.ndarray:
        """ Computes the theoretical bandpowers for given parameters using JAX for Just-In-Time (JIT) compilation.

        This function leverages JAX's JIT compilation to optimize the computation of theoretical bandpowers
        based on input parameters. It uses predefined constants for galaxy clustering (GC) and weak lensing (WL)
        number densities and bandpower matrices.

        Args:
            parameters (np.ndarray): An array of cosmological parameters.

        Returns:
            np.ndarray: The computed theoretical bandpowers.
        """
        return get_bandpowers_theory(parameters, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL)

    def planck_flow_like(cosmology: np.ndarray) -> float:
        """
        Calculates the log-likelihood for a given cosmology using a flow-based likelihood function.

        This function evaluates the log-likelihood of the given cosmological parameters using the
        flow-based likelihood (FLOW).

        Args:
            cosmology (np.ndarray): An array of cosmological parameters.

        Returns:
            float: The log-likelihood value.
        """
        flowlike = FLOW.loglike(cosmology).item()
        return flowlike

    def planck_exact_like(cosmology: np.ndarray) -> float:
        """
        Computes the exact log-likelihood for a given cosmology using Planck data.

        This function evaluates the log-likelihood of the given cosmological parameters using an exact
        likelihood function for Planck data.

        Args:
            cosmology (np.ndarray): An array of cosmological parameters.

        Returns:
            float: The log-likelihood value.
        """
        return plite_loglike(likelihood, cosmology, cfg)

    def DESY1_like(parameters: np.ndarray) -> float:
        """
        Computes the log-likelihood for the DESY1 dataset given cosmological parameters.

        This function calculates the log-likelihood by comparing the observed data with the theoretical
        predictions based on the provided cosmological parameters. It uses precomputed data, precision
        matrices, and JAX-compiled theory calculations.

        Args:
            parameters (np.ndarray): An array of cosmological parameters.

        Returns:
            float: The log-likelihood value.
        """
        theory = jit_theory(parameters)
        diff = DATA - theory
        chi2 = diff @ PRECISION @ diff
        logl = -0.5 * jnp.nan_to_num(chi2, nan=np.inf, posinf=np.inf, neginf=np.inf).item()
        return logl

    def joint_like(sigma8, omegac, omegab, hubble, ns, m1, m2, m3, m4,
                    dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4, a_ia, eta,
                    b1, b2, b3, b4, b5, dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5) -> float:

        """
        Computes the joint log-likelihood for DESY1 and Planck data given cosmological and nuisance parameters.

        This function calculates the combined log-likelihood of DESY1 and Planck datasets based on the input
        cosmological and nuisance parameters. It constructs the parameter arrays for DESY1 and Planck likelihood
        calculations, evaluates each likelihood, and returns their sum.

        Args:
            sigma8 (float): Amplitude of matter fluctuations on 8 Mpc/h scales.
            omegac (float): Cold dark matter density parameter.
            omegab (float): Baryon density parameter.
            hubble (float): Dimensionless Hubble parameter (H0 / 100).
            ns (float): Scalar spectral index.
            m1, m2, m3, m4 (float): multiplicative nuisance parameters in cosmic shear.
            dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4 (float): Photometric redshift bias parameters for weak lensing.
            a_ia (float): Amplitude of intrinsic alignments.
            eta (float): Slope parameter for intrinsic alignments.
            b1, b2, b3, b4, b5 (float): Galaxy bias parameters.
            dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5 (float): Photometric redshift bias parameters for galaxy clustering.

        Returns:
            float: The combined log-likelihood value of DESY1 and Planck datasets.
        """
        parameters = jnp.array([sigma8, omegac, omegab, hubble, ns, m1, m2, m3, m4,
                                dz_wl_1, dz_wl_2, dz_wl_3, dz_wl_4, a_ia, eta,
                                b1, b2, b3, b4, b5, dz_gc_1, dz_gc_2, dz_gc_3, dz_gc_4, dz_gc_5])

        # the set of cosmological parameters
        cosmology = np.array([sigma8, omegac, omegab, hubble, ns])

        # the DES log-likelihood
        des_like = DESY1_like(parameters)

        # the Planck log-likelihood calculation (exact or flow)
        if cfg.useflow:
            planck_like = planck_flow_like(cosmology)
        else:
            planck_like = planck_exact_like(cosmology)
        return des_like + planck_like

    # Set up the input
    create_experiment_path(cfg)
    info = {"likelihood": {"des_planck": joint_like}}
    info["params"] = get_params_info_des(cfg)
    info["sampler"] = {
        "mcmc": {
            "max_samples": cfg.nsamples,
            "Rminus1_stop": 0.01,
        }
    }
    info["output"] = os.path.join(cfg.output_folder, cfg.output_name, cfg.output_name)

    # normal Python run
    updated_info, sampler = run(info)
    return sampler

if __name__ == "__main__":
    sampler = SampleJoint()
