"""
Code: Planck Lite likelihood code.
Date: August 2024
Author: Arrykrishna
"""

import os
import logging
import numpy as np
import camb

# Prince's code
from planck.plite import PlanckLitePy
from planck.params import PlanckCosmo
from cfglib import DESY1PlanckConfig

LOGGER = logging.getLogger(os.path.basename(__file__))


def planck_theory(parameters: PlanckCosmo, cfg: DESY1PlanckConfig) -> dict:
    """
    Calculate the CMB power spectra using CAMB.

    Args:
        parameters (PlanckCosmo): an object of parameters.
        cfg (DESY1PlanckConfig): the main configuration file
    Returns:
        dict: a dictionary with the power spectra and the ells.
    """
    pars = camb.set_params(
        H0=parameters.H0,
        ombh2=parameters.ombh2,
        omch2=parameters.omch2,
        ns=parameters.ns,
        As=cfg.fid_as,
    )
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    s8_fid = results.get_sigma8_0()
    pars.InitPower.set_params(
        As=cfg.fid_as * parameters.sigma8**2 / s8_fid**2, ns=parameters.ns
    )
    pars.set_for_lmax(cfg.ellmax, lens_potential_accuracy=cfg.lens_potential_accuracy)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")

    # The different CL are always in the order TT, EE, BB, TE
    camb_tt = powers[cfg.spectratype][:, 0]
    camb_ee = powers[cfg.spectratype][:, 1]
    camb_te = powers[cfg.spectratype][:, 3]

    ells = np.arange(camb_tt.shape[0])
    condition = (ells >= 2) & (ells <= 2508)

    powerspectra = {
        "ells": ells[condition],
        "tt": camb_tt[condition],
        "te": camb_te[condition],
        "ee": camb_ee[condition],
    }

    return powerspectra


def plite_loglike(
    likelihood: PlanckLitePy, parameter: np.ndarray, cfg: DESY1PlanckConfig
) -> np.ndarray:
    """
    Calculate the log-likelihood given a set of points.

    Args:
        parameter (np.ndarray): a set of points of dimension d
        cfg (ConfigDict): the main configuration file

    Returns:
        np.ndarray: the log-likelihood values.
    """
    point = PlanckCosmo(
        sigma8=parameter[0],
        Omega_c=parameter[1],
        Omega_b=parameter[2],
        h=parameter[3],
        ns=parameter[4],
    )
    cls = planck_theory(point, cfg)
    loglike = likelihood.loglike(cls["tt"], cls["te"], cls["ee"], min(cls["ells"]))
    LOGGER.info(f"log-likelihood is {loglike:.3f}")
    return loglike
