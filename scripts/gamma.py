import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_gamma(x, y, out, title, ylabel):
    # Create a smooth curve using spline interpolation
    spline = make_interp_spline(x, y, k=3)  # Cubic spline interpolation
    smooth_x = np.linspace(min(x), max(x), 100)
    smooth_y = spline(smooth_x)

    # Plot the smooth curve
    plt.plot(smooth_x, smooth_y, linestyle="--", color="gray", label="Trend Curve")
    # Plot the original data points with different labels
    plt.scatter(x[0], y[0], marker="^", color="red", label="FedAvg")
    plt.scatter(x[1:], y[1:], marker="^", color="blue", label="FedTTS")

    # Set plot title
    plt.title(title)

    # Set axis labels
    plt.xlabel(r"$\gamma$")
    plt.ylabel(ylabel)

    # Show legend
    plt.legend()

    # Show the plot
    print(f"Save file {out}")
    plt.savefig(out, dpi=300)
    plt.close()


if __name__ == "__main__":
    # Data
    gammas = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    rounds = [135, 109, 131, 113, 123, 109]
    times = [777.2058, 587.2055, 797.1033, 529.5589, 578.9952, 505.2526]
    plot_gamma(
        gammas,
        rounds,
        "save/gamma-rounds.png",
        r"Communication Rounds varying with $\gamma$ [3 Clusters]",
        "Communication Rounds",
    )
    plot_gamma(
        gammas,
        times,
        "save/gamma-times.png",
        r"Running Time Varying with $\gamma$ [3 Clusters]",
        "Seconds",
    )
