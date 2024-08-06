from omegaconf import MISSING
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FlowConfig:
    fname: str = MISSING
    nsamples: Optional[int] = None
    lr: float = 5e-3
    nsteps: int = 10


@dataclass
class SamplerConfig:
    nmcmc: int = MISSING
    joint: List[str] = MISSING


@dataclass
class DESY1PlanckConfig:

    # We use EMCEE to sample the joint flows
    nmcmc: int = 50
    eps: float = 1e-3
    mcmc_fname: str = "desy1_flow_planck_flow"

    # camb setting
    lens_potential_accuracy: int = 2
    omega_k: float = 0.0
    tau: float = 0.054
    ellmax: int = 4000
    fid_as: float = 2.1e-9

    # DES Y1 setting
    use_emu: bool = False

    # cobaya setting
    proposal: float = 1e-3
    output_folder: str = "/mnt/users/phys2286/projects/emuflow/DESPlanck/"
    output_name: str = 'testing'
    nsamples = 10

    # planck setting
    planckdata: str = "planck/data"
    spectra: str = "TTTEEE"
    spectratype: str = "total"
    use_low_ell_bins: bool = True
    year: int = 2018

    # which flow to use
    useflow: bool = False
    flow_name: str = 'base_plikHM_TTTEEE_lowl_lowE'
