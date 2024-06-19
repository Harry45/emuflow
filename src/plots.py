import os
import numpy as np
import matplotlib.pylab as plt
from getdist import plots, MCSamples
from hydra.core.hydra_config import HydraConfig

plt.rc("text", usetex=True)
plt.rc("font", **{"family": "sans-serif", "serif": ["Palatino"]})
FONTSIZE = 20


def plot_loss(loss: list, fname: str):
    """Plot the loss function when training the normalising flow.

    Args:
        loss (list): the loss values
        fname (str): name of the file
    """
    folder = HydraConfig.get()["runtime"]["output_dir"]
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(len(loss)) + 1, loss, lw=2, c="r")
    plt.ylabel(r"$\mathcal{L}$", fontsize=FONTSIZE)
    plt.xlabel(r"$i$", fontsize=FONTSIZE)
    plt.tick_params(axis="x", labelsize=FONTSIZE)
    plt.tick_params(axis="y", labelsize=FONTSIZE)
    plt.savefig(f"{folder}/loss_{fname}.pdf", bbox_inches="tight")
    plt.savefig(f"{folder}/loss_{fname}.png", bbox_inches="tight")
    plt.close()


def triangle_cosmology(
    samples1: np.ndarray, samples2: np.ndarray, label1: str, label2: str, fname: str
):
    """Creates a triangle plot given the original samples and the flow samples.

    Args:
        samples1 (np.ndarray): the original samples
        samples2 (np.ndarray): the flow samples
        label1 (str): label for the first set of samples
        label2 (str): label for the second set of samples
        fname (str): name of the file for output
    """

    settings = {
        "mult_bias_correction_order": 0,
        "smooth_scale_2D": 0.3,
        "smooth_scale_1D": 0.3,
    }
    color2 = "#50C878"
    color1 = "#222E50"
    folder = HydraConfig.get()["runtime"]["output_dir"]

    ndim = 5
    names = ["x%s" % i for i in range(ndim)]
    labels = [r"$\sigma_{8}$", r"$\Omega_{c}$", r"$\Omega_{b}$", r"$h$", r"$n_{s}$"]

    samples_1 = MCSamples(
        samples=samples1, names=names, labels=labels, label=label1, settings=settings
    )
    samples_2 = MCSamples(
        samples=samples2, names=names, labels=labels, label=label2, settings=settings
    )

    G = plots.getSubplotPlotter(subplot_size=1.0)
    G.settings.solid_contour_palefactor = 0.9
    G.settings.alpha_filled_add = 0.9
    G.settings.num_plot_contours = 2
    G.settings.lw_contour = 1
    G.settings.axes_fontsize = 15
    G.settings.lab_fontsize = 15
    G.settings.fontsize = 35
    G.settings.legend_fontsize = 14
    samples_1.updateSettings({"contours": [0.68, 0.95]})
    samples_2.updateSettings({"contoburs": [0.68, 0.95]})
    G.triangle_plot(
        [samples_2, samples_1],
        filled=[True, False],
        contour_colors=[color2, color1],
        contour_lws=[2, 2],
        contour_ls=["-", "-"],
        legend_loc=(0.45, 0.88),
    )
    plotname = f"{folder}/triangle_{fname}"
    plt.savefig(f"{plotname}.pdf", transparent=False, bbox_inches="tight")
    plt.savefig(f"{plotname}.png", transparent=False, bbox_inches="tight")
    plt.close()
