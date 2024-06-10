import os
import logging
import torch
import numpy as np
from emcee import EnsembleSampler
from hydra.utils import get_original_cwd
from src.flow import NormFlow

LOGGER = logging.getLogger(os.path.basename(__file__))


def load_flow(experiment: str) -> NormFlow:
    """Load a pre-trained normalising flow.

    Args:
        experiment (str): name of the experiment (flow)

    Returns:
        NormFlow: the pre-trained normalising flow
    """
    path = get_original_cwd()
    fullpath = os.path.join(path, f"flows/{experiment}.pt")
    flow = torch.load(fullpath)
    return flow


def loglike_flows(sample: np.ndarray, flows: dict) -> float:
    """The log-likelihood calculated by summing the log-probability from each flow.

    Args:
        sample (np.ndarray): a sample in parameter space
        flows (dict): a dictionary of flows

    Returns:
        float: the log-likelihood value
    """
    keys = list(flows.keys())
    logl = sum([flows[key].loglike(sample) for key in keys])
    return logl


class SampleFlow:
    """Draw samples from the joint, given a list of experiments.

    Args:
        experiments (list): a list of experiments we want to use.
    """

    def __init__(self, experiments: list):

        self.flows = {e: load_flow(e) for e in experiments}
        self.nexp = len(self.flows)
        self.mean = self.flows[experiments[0]].samples.mean(0)
        self.ndim = len(self.mean)
        self.nwalkers = 2 * self.ndim

        LOGGER.info(f"Number of flows/experiments is: {self.nexp}")
        LOGGER.info(f"Number of dimensions is : {self.ndim}")
        LOGGER.info(f"Number of walkers is : {self.nwalkers}")

    def sampler(self, nsamples: int, eps: float) -> EnsembleSampler:
        """Sample the joint using emcee.

        Args:
            nsamples (int): the number of samples we want
            eps (float): the step-size to use in emcee.

        Returns:
            EnsembleSampler: the emcee Ensemble sampler
        """
        LOGGER.info(f"Number of MCMC samples per walker is : {nsamples}")
        pos = self.mean + eps * np.random.randn(self.nwalkers, self.ndim)
        sampler = EnsembleSampler(
            self.nwalkers, self.ndim, loglike_flows, args=(self.flows,)
        )
        sampler.run_mcmc(pos, nsamples, progress=True)
        return sampler
