import matplotlib.pyplot as plt
from cycler import cycler
from tueplots import bundles
from tueplots.constants.color import palettes
# from tueplots import cycler


def setup_plotting():
    """
    Configure matplotlib to match the LaTeX report style using tueplots.
    Ensures consistent font sizes, spacing, and figure aesthetics.
    """

    # Apply tueplots bundle (layout, sizes, spacing)
    plt.rcParams.update(bundles.icml2022())

    # Disable LaTeX rendering for notebooks (avoids SVG / LaTeX errors)
    plt.rcParams["text.usetex"] = False

    # Define a custom color cycle (TU red)
    plt.rcParams["axes.prop_cycle"] = cycler(color=["#a51e36"])

    # optional plotting cycler from tue_plot
    #plt.rcParams.update(cycler.cycler(color=palettes.tue_plot))
