import os
import shutil
from cobaya.run import run
from cobaya.model import get_model
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import sacc
import torch
from src.flow import NormFlow


# our script
from cosmology.bandpowers import get_bandpowers_theory
from cosmology.bandpowers import (
    get_nz,
    scale_cuts,
    extract_bandwindow,
    extract_data_covariance,
)
import jax_cosmo as jc
from src.utils import pickle_load
from jax.lib import xla_bridge

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".75"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

print(xla_bridge.get_backend().platform)

# setting up cobaya, jaxcosmo and emulator
# jax.config.update("jax_default_device", jax.devices("cpu")[0])
jc.power.USE_EMU = False
PROPOSAL = 1e-3
NSAMPLES = 500000
MAIN_PATH = "./"  # "/mnt/zfsusers/phys2286/projects/DESEMU/"
OUTPUT_FOLDER = MAIN_PATH + "DESPlanck/mcmc_1/"

if os.path.exists(OUTPUT_FOLDER) and os.path.isdir(OUTPUT_FOLDER):
    shutil.rmtree(OUTPUT_FOLDER)
os.makedirs(OUTPUT_FOLDER)


def load_flow(experiment: str) -> NormFlow:
    """Load a pre-trained normalising flow.

    Args:
        experiment (str): name of the experiment (flow)

    Returns:
        NormFlow: the pre-trained normalising flow
    """
    fullpath = f"flows/{experiment}.pt"
    flow = torch.load(fullpath)
    return flow


def load_data(fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000):
    saccfile = sacc.Sacc.load_fits(f"data/{fname}.fits")
    jax_nz_wl = get_nz(saccfile, tracertype="wl")
    jax_nz_gc = get_nz(saccfile, tracertype="gc")
    saccfile_cut = scale_cuts(saccfile, kmax=kmax, lmin_wl=lmin_wl, lmax_wl=lmax_wl)
    print("Loaded data")
    bw_gc, bw_gc_wl, bw_wl = extract_bandwindow(saccfile_cut)
    data, covariance = extract_data_covariance(saccfile_cut)
    newcov = covariance + jnp.eye(data.shape[0]) * 1e-18
    precision = np.linalg.inv(np.asarray(newcov))
    precision = jnp.asarray(precision)
    return data, precision, jax_nz_gc, jax_nz_wl, bw_gc, bw_gc_wl, bw_wl


DATA, PRECISION, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL = load_data(
    fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000
)
FLOW = load_flow("base_plikHM_TTTEEE_lowl_lowE")


@jax.jit
def jit_theory(parameters):
    return get_bandpowers_theory(parameters, NZ_GC, NZ_WL, BW_GC, BW_GC_WL, BW_WL)


def cobaya_logl(
    sigma8,
    omegac,
    omegab,
    hubble,
    ns,
    m1,
    m2,
    m3,
    m4,
    dz_wl_1,
    dz_wl_2,
    dz_wl_3,
    dz_wl_4,
    a_ia,
    eta,
    b1,
    b2,
    b3,
    b4,
    b5,
    dz_gc_1,
    dz_gc_2,
    dz_gc_3,
    dz_gc_4,
    dz_gc_5,
):
    parameters = jnp.array(
        [
            sigma8,
            omegac,
            omegab,
            hubble,
            ns,
            m1,
            m2,
            m3,
            m4,
            dz_wl_1,
            dz_wl_2,
            dz_wl_3,
            dz_wl_4,
            a_ia,
            eta,
            b1,
            b2,
            b3,
            b4,
            b5,
            dz_gc_1,
            dz_gc_2,
            dz_gc_3,
            dz_gc_4,
            dz_gc_5,
        ]
    )
    theory = jit_theory(parameters)
    diff = DATA - theory
    chi2 = diff @ PRECISION @ diff
    logl = -0.5 * jnp.nan_to_num(chi2, nan=np.inf, posinf=np.inf, neginf=np.inf).item()

    # contribution due to the flow (we can ignore the prior since it is uniform)
    cosmology = np.array([sigma8, omegac, omegab, hubble, ns])
    flowl = FLOW.loglike(cosmology).item()
    return logl + flowl


# Set up the input
info = {"likelihood": {"my_likelihood": cobaya_logl}}
info["params"] = {
    # cosmological parameters
    "sigma8": {
        "prior": {"min": 0.60, "max": 1.0},
        "ref": {"dist": "norm", "loc": 0.85, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegac": {
        "prior": {"min": 0.14, "max": 0.35},
        "ref": {"dist": "norm", "loc": 0.25, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "omegab": {
        "prior": {"min": 0.03, "max": 0.055},
        "ref": {"dist": "norm", "loc": 0.04, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "hubble": {
        "prior": {"min": 0.64, "max": 0.82},
        "ref": {"dist": "norm", "loc": 0.70, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "ns": {
        "prior": {"min": 0.87, "max": 1.07},
        "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    # multiplicative bias parameters
    "m1": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m2": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m3": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "m4": {
        "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
        "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    # shifts weak lensing bins
    "dz_wl_1": {
        "prior": {"dist": "norm", "loc": -0.001, "scale": 0.016},
        "ref": {"dist": "norm", "loc": -0.001, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_2": {
        "prior": {"dist": "norm", "loc": -0.019, "scale": 0.013},
        "ref": {"dist": "norm", "loc": -0.019, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_3": {
        "prior": {"dist": "norm", "loc": 0.009, "scale": 0.011},
        "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_wl_4": {
        "prior": {"dist": "norm", "loc": -0.018, "scale": 0.022},
        "ref": {"dist": "norm", "loc": -0.018, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    # intrinsic alignment
    "a_ia": {
        "prior": {"min": -1.0, "max": 1.0},
        "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "eta": {
        "prior": {"min": -5.0, "max": 5.0},
        "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.0001},
        "proposal": PROPOSAL,
    },
    # multiplicative bias (galaxy clustering)
    "b1": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.34, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b2": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.57, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b3": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.59, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b4": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    "b5": {
        "prior": {"min": 0.8, "max": 3.0},
        "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
        "proposal": PROPOSAL,
    },
    # shifts (galaxy clustering)
    "dz_gc_1": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_2": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
        "ref": {"dist": "norm", "loc": -0.0015, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_3": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.006},
        "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_4": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
        "proposal": PROPOSAL,
    },
    "dz_gc_5": {
        "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
        "ref": {"dist": "norm", "loc": -0.012, "scale": 0.001},
        "proposal": PROPOSAL,
    },
}

pnames = list(info["params"].keys())
covmat = pickle_load("data", "cov_jaxcosmo")
info["sampler"] = {
    "mcmc": {
        "max_samples": NSAMPLES,
        "Rminus1_stop": 0.01,
        "covmat": covmat,
        "covmat_params": pnames,
    }
}
info["output"] = OUTPUT_FOLDER + "des_planck"

# normal Python run
updated_info, sampler = run(info)
