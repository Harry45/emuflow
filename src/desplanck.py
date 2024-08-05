import sacc
import jax.numpy as jnp
import numpy as np
from cosmology.bandpowers import (
    get_nz,
    scale_cuts,
    extract_bandwindow,
    extract_data_covariance,
)

def load_data(fname="cls_DESY1", kmax=0.15, lmin_wl=30, lmax_wl=2000):
    """
    Loads and processes cosmological data from a SACC file for DESY1 analysis.

    This function loads a SACC file containing cosmological data, extracts number density
    distributions for weak lensing (wl) and galaxy clustering (gc), applies scale cuts
    to the data, and extracts bandpower windows, data vectors, and the covariance matrix.
    It then computes a regularized precision matrix.

    Args:
        fname (str, optional): Filename of the SACC file (without extension). Defaults to "cls_DESY1".
        kmax (float, optional): Maximum wavenumber for scale cuts. Defaults to 0.15.
        lmin_wl (int, optional): Minimum multipole for weak lensing scale cuts. Defaults to 30.
        lmax_wl (int, optional): Maximum multipole for weak lensing scale cuts. Defaults to 2000.

    Returns:
        tuple: A tuple containing the following elements:
            - data (jnp.ndarray): Data vector extracted from the SACC file.
            - precision (jnp.ndarray): Regularized precision matrix.
            - jax_nz_gc (Any): Number density distribution for galaxy clustering.
            - jax_nz_wl (Any): Number density distribution for weak lensing.
            - bw_gc (Any): Bandpower window for galaxy clustering.
            - bw_gc_wl (Any): Bandpower window for galaxy clustering and weak lensing.
            - bw_wl (Any): Bandpower window for weak lensing.
    """
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

def get_params_info_des(cfg):
    parameters = {
        # cosmological parameters
        "sigma8": {
            "prior": {"min": 0.60, "max": 1.0},
            "ref": {"dist": "norm", "loc": 0.85, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "omegac": {
            "prior": {"min": 0.14, "max": 0.35},
            "ref": {"dist": "norm", "loc": 0.25, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "omegab": {
            "prior": {"min": 0.03, "max": 0.055},
            "ref": {"dist": "norm", "loc": 0.04, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "hubble": {
            "prior": {"min": 0.64, "max": 0.82},
            "ref": {"dist": "norm", "loc": 0.70, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "ns": {
            "prior": {"min": 0.87, "max": 1.07},
            "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        # multiplicative bias parameters
        "m1": {
            "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "m2": {
            "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "m3": {
            "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "m4": {
            "prior": {"dist": "norm", "loc": 0.012, "scale": 0.023},
            "ref": {"dist": "norm", "loc": 0.012, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        # shifts weak lensing bins
        "dz_wl_1": {
            "prior": {"dist": "norm", "loc": -0.001, "scale": 0.016},
            "ref": {"dist": "norm", "loc": -0.001, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_wl_2": {
            "prior": {"dist": "norm", "loc": -0.019, "scale": 0.013},
            "ref": {"dist": "norm", "loc": -0.019, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_wl_3": {
            "prior": {"dist": "norm", "loc": 0.009, "scale": 0.011},
            "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_wl_4": {
            "prior": {"dist": "norm", "loc": -0.018, "scale": 0.022},
            "ref": {"dist": "norm", "loc": -0.018, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        # intrinsic alignment
        "a_ia": {
            "prior": {"min": -1.0, "max": 1.0},
            "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "eta": {
            "prior": {"min": -5.0, "max": 5.0},
            "ref": 0.0,  # {"dist": "norm", "loc": 0.0, "scale": 0.0001},
            "proposal": cfg.proposal,
        },
        # multiplicative bias (galaxy clustering)
        "b1": {
            "prior": {"min": 0.8, "max": 3.0},
            "ref": {"dist": "norm", "loc": 1.34, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "b2": {
            "prior": {"min": 0.8, "max": 3.0},
            "ref": {"dist": "norm", "loc": 1.57, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "b3": {
            "prior": {"min": 0.8, "max": 3.0},
            "ref": {"dist": "norm", "loc": 1.59, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "b4": {
            "prior": {"min": 0.8, "max": 3.0},
            "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "b5": {
            "prior": {"min": 0.8, "max": 3.0},
            "ref": {"dist": "norm", "loc": 1.9, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        # shifts (galaxy clustering)
        "dz_gc_1": {
            "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
            "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_gc_2": {
            "prior": {"dist": "norm", "loc": 0.0, "scale": 0.007},
            "ref": {"dist": "norm", "loc": -0.0015, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_gc_3": {
            "prior": {"dist": "norm", "loc": 0.0, "scale": 0.006},
            "ref": {"dist": "norm", "loc": 0.02, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_gc_4": {
            "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
            "ref": {"dist": "norm", "loc": 0.009, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "dz_gc_5": {
            "prior": {"dist": "norm", "loc": 0.0, "scale": 0.01},
            "ref": {"dist": "norm", "loc": -0.012, "scale": 0.001},
            "proposal": cfg.proposal,
        },
    }
    return parameters

def get_params_info_planck(cfg):
    parameters = {
        # cosmological parameters
        "sigma8": {
            "prior": {"min": 0.60, "max": 1.0},
            "ref": {"dist": "norm", "loc": 0.85, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "omegac": {
            "prior": {"min": 0.14, "max": 0.35},
            "ref": {"dist": "norm", "loc": 0.25, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "omegab": {
            "prior": {"min": 0.03, "max": 0.055},
            "ref": {"dist": "norm", "loc": 0.04, "scale": 0.001},
            "proposal": cfg.proposal,
        },
        "hubble": {
            "prior": {"min": 0.64, "max": 0.82},
            "ref": {"dist": "norm", "loc": 0.70, "scale": 0.01},
            "proposal": cfg.proposal,
        },
        "ns": {
            "prior": {"min": 0.87, "max": 1.07},
            "ref": {"dist": "norm", "loc": 1.0, "scale": 0.01},
            "proposal": cfg.proposal,
        },
    }
    return parameters