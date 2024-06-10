import os
import torch
import tqdm
import random
import logging
import numpy as np
import pandas as pd
from typing import Tuple
import torch.nn as nn
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import flowtorch.parameters as params
from hydra.core.hydra_config import HydraConfig
from src.plots import triangle_cosmology, plot_loss
from cfglib import FlowConfig

LOGGER = logging.getLogger(os.path.basename(__file__))


def create_dataset(
    samples: np.ndarray, nsamples: int = None
) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
    """Creates a dataset with the samples, mean of the samples
    and the standard deviation of the samples. We can also specify
    the number of samples to use and nsamples are chosen randomly
    from the original samples.

    Args:
        samples (np.ndarray): the samples
        nsamples (int, optional): the number of samples to use. Defaults to None.

    Returns:
        Tuple[torch.tensor, torch.tensor, torch.tensor]: the samples, mean and its standard deviation
    """
    mean_np = np.median(samples, axis=0)
    std_np = np.std(samples, axis=0)

    if nsamples is not None:
        nsamples_original = samples.shape[0]
        idx = random.sample(range(1, nsamples_original), nsamples)
        samples = samples[idx]

    dataset = torch.tensor(samples, dtype=torch.float)
    mean = torch.tensor(mean_np, dtype=torch.float)
    std = torch.tensor(std_np, dtype=torch.float)
    return dataset, mean, std


class SampleLoader:
    def __init__(self, fname: str):
        """Load the cosmological samples (5 parameters).

        Args:
            fname (str): name of the file.
        """
        df = pd.read_csv(f"samples/{fname}.csv", index_col=0)
        self.samples = df.values
        self.columns = list(df.columns)
        self.name = fname
        LOGGER.info(f"Parameters of the model : {self.columns}")

    def __str__(self):
        return self.name


class TransForm:
    """Pre-whitens the input parameters using the full set of samples.
    The samples are centred on zero and the covariance is the identity
    matrix.

    Args:
        samples (np.ndarray): the MCMC samples
    """

    def __init__(self, samples: np.ndarray):
        self.ndim = samples.shape[1]
        self.samples = samples
        self.cov_train = np.cov(samples.T)
        self.cholesky = np.linalg.cholesky(self.cov_train)
        self.cholesky_inv = np.linalg.inv(self.cholesky)
        self.mean = np.mean(samples, axis=0).reshape(-1, self.ndim)

    def forward(self, testpoint: np.ndarray) -> np.ndarray:
        """Scale the test point by the mean and the cholesky factor.

        Args:
            testpoint (np.ndarray): the test point in parameter space.

        Returns:
            np.ndarray: the transformed point.
        """
        testpoint = testpoint.reshape(-1, self.ndim)
        testtrans = self.cholesky_inv @ (testpoint - self.mean).T
        return testtrans.T

    def inverse(self, prediction: np.ndarray) -> np.ndarray:
        """Apply the inverse transformation on the prediction.

        Args:
            prediction (np.ndarray): the sampled point from the normalising flow.

        Returns:
            np.ndarray: the transformed point.
        """
        prediction = prediction.reshape(-1, self.ndim)
        predtrans = self.cholesky @ prediction.T + self.mean.T
        return predtrans.T


def print_message(max_samples: int):
    """Print message for user to choose number of samples in the flow.

    Args:
        max_samples (int): The maximum number of samples.
    """
    print("=" * 75)
    print("A maximum of 15000 should suffice in 5D.")
    print(f"nsamples should be < {max_samples}")
    print("=" * 75)


def build_network(hidden=(32, 32, 32)):
    """Creates a function to do the mapping

    Args:
        hidden (tuple, optional): Number of hidden layers. Defaults to (32, 32, 32).

    Returns:
        flowtorch.lazy.lazy: a composition of bijectors
    """

    transforms = bij.Compose(
        bijectors=[
            bij.AffineAutoregressive(
                params.DenseAutoregressive(hidden_dims=hidden, nonlinearity=nn.Tanh),
            ),
            bij.AffineAutoregressive(
                params.DenseAutoregressive(hidden_dims=hidden, nonlinearity=nn.Tanh),
            ),
            bij.AffineAutoregressive(
                params.DenseAutoregressive(hidden_dims=hidden, nonlinearity=nn.Tanh),
            ),
        ]
    )
    return transforms


class NormFlow(TransForm, SampleLoader):
    """Build a normalizing flow on the MCMC samples.

    Args:
        experiment (str): the name of the experiment
        nsamples (int, optional): the number of samples. Defaults to None.
    """

    def __init__(self, experiment: str, nsamples: int = None):

        SampleLoader.__init__(self, experiment)
        print_message(self.samples.shape[0])

        TransForm.__init__(self, self.samples)
        data = TransForm.forward(self, self.samples)
        self.dataset, self.fiducial, self.std = create_dataset(data, nsamples)

        dist_x = torch.distributions.Independent(
            torch.distributions.Normal(self.fiducial, self.std), 1
        )
        bijector = build_network(hidden=(16, 16, 16))
        self.dist_y = dist.Flow(dist_x, bijector)

        LOGGER.info(f"Total number of samples available is :{self.samples.shape[0]}")
        LOGGER.info(f"Number of samples in NF is : {self.dataset.shape[0]}")

    def training(self, lr: float = 5e-3, nsteps: int = 1000) -> list:
        """Train the normalising flow

        Args:
            lr (float, optional): the learning rate. Defaults to 5e-3.
            nsteps (int, optional): the number of steps. Defaults to 1000.

        Returns:
            list: a list of the loss values at each iteration.
        """
        optimizer = torch.optim.Adam(self.dist_y.parameters(), lr=lr)
        record = []
        interval = divmod(nsteps, 20)[0]
        with tqdm.trange(nsteps) as bar:
            for step in bar:
                optimizer.zero_grad()
                loss = -self.dist_y.log_prob(self.dataset).mean()
                loss.backward()
                optimizer.step()
                record.append(loss.item())
                postfix = dict(Loss=f"{loss.item():.3f}")
                bar.set_postfix(postfix)
                if step % interval == 0:
                    LOGGER.info(str(bar))
            LOGGER.info(f"Final loss value: {loss.item():.3f}")
        return record

    def generate_samples(self, nsamples: int) -> np.ndarray:
        """Generate samples from the normalizing flow.

        Args:
            nsamples (int): the number of samples we want

        Returns:
            np.ndarray: the samples from the normalising flow.
        """
        size = torch.Size(
            [
                nsamples,
            ]
        )

        gen_samples = self.dist_y.sample(size).detach().numpy()
        gen_samples = TransForm.inverse(self, gen_samples)
        return gen_samples

    def loglike(self, parameter: np.ndarray) -> np.ndarray:
        """Calculates the log-probability of the flow given a sample.

        Args:
            parameter (np.ndarray): a test point in Cosmological parameter space.

        Returns:
            np.ndarray: the log-probability value.
        """
        parameter = TransForm.forward(self, parameter)
        p_tensor = torch.tensor(parameter, dtype=torch.float)
        vol_correction = np.log(np.linalg.det(self.cholesky_inv))
        return self.dist_y.log_prob(p_tensor).detach().numpy() + vol_correction


def save_flow(flow: NormFlow, fname: str):
    folder = HydraConfig.get()["runtime"]["output_dir"]
    torch.save(flow, f"{folder}/{fname}.pt")


def build_flow(cfg: FlowConfig):

    # train the flow
    flow = NormFlow(cfg.fname, cfg.nsamples)
    loss = flow.training(lr=cfg.lr, nsteps=cfg.nsteps)
    save_flow(flow, cfg.fname)

    # generate a triangle plot to check the flow performance
    if cfg.plot.genplot:
        # plot the loss function
        plot_loss(loss, cfg.fname)

        flow_samples = flow.generate_samples(cfg.plot.nfsamples)
        triangle_cosmology(
            flow.samples,
            flow_samples,
            cfg.plot.label1,
            cfg.plot.label2,
            cfg.fname,
        )
    return flow
