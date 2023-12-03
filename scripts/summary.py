import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


def plot_cluster_info(
    avg_time, tts_time, cluster_labels, title, xlabel, ylabel, outfile, bar_width=0.35
):
    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    k = cluster_labels.shape[0] - 1
    # Plot histograms for avg_time and tts_time in different clusters

    avg_bar = ax.bar(
        cluster_labels - bar_width / 2,
        avg_time,
        bar_width,
        color="blue",
        alpha=0.7,
        label="FedAvg",
    )
    tts_bar = ax.bar(
        cluster_labels + bar_width / 2,
        tts_time,
        bar_width,
        color="orange",
        alpha=0.7,
        label="FedTTS",
    )

    # Get the top points (maximum heights) of each histogram bar
    avg_max_points = [rect.get_height() for rect in avg_bar]
    tts_max_points = [rect.get_height() for rect in tts_bar]

    # Create smooth curves using spline interpolation
    spline_avg = make_interp_spline(cluster_labels - bar_width / 2, avg_max_points, k=k)
    spline_tts = make_interp_spline(cluster_labels + bar_width / 2, tts_max_points, k=k)

    # Plot smooth curves
    cluster_labels_smooth = np.linspace(cluster_labels.min(), cluster_labels.max(), 100)
    ax.plot(
        cluster_labels_smooth,
        spline_avg(cluster_labels_smooth),
        color="blue",
    )
    ax.plot(
        cluster_labels_smooth,
        spline_tts(cluster_labels_smooth),
        color="orange",
    )

    # Set plot title
    plt.title(title)

    # Set axis labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Show legend
    plt.legend()

    # Show the plot
    print(f"Save file to {outfile}")
    plt.savefig(outfile, dpi=300)
    plt.close()


def his_cluster():
    avg_time = [777.2068, 1666.8982, 4956.2813]
    tts_time = [587.2055, 1230.4804, 3751.6850]
    labels = np.array([1, 2, 3])
    # plot_cluster_time(avg_time, tts_time)
    plot_cluster_info(
        avg_time,
        tts_time,
        labels,
        "Running Time with Different Clusters",
        "Number of Clusters",
        "Running Time (seconds)",
        "save/cluster-time-his.png",
    )
    avg_rounds = [135, 315, 719]
    tts_rounds = [109, 189, 525]
    plot_cluster_info(
        avg_rounds,
        tts_rounds,
        labels,
        "Communication Rounds with Different Clusters",
        "Number of Clusters",
        "Rounds",
        "save/cluster-rounds-his.png",
    )


def his_overlapping():
    # avg_time = [595.4625, 570.4938, 541.3340]
    # tts_time = [556.520, 670.4787, 500.256]
    labels = np.array([0.2, 0.5, 0.8])
    # # plot_cluster_time(avg_time, tts_time)
    # plot_cluster_info(
    #     avg_time,
    #     tts_time,
    #     labels,
    #     "Running Time with Different Overlapping Rate [3 Clusters]",
    #     "Overlapping Rate Among Clusters",
    #     "Running Time (seconds)",
    #     "save/over-time-his.png",
    #     0.05,
    # )
    avg_rounds = [127, 123, 111]
    tts_rounds = [123, 119, 101]
    plot_cluster_info(
        avg_rounds,
        tts_rounds,
        labels,
        "Communication Rounds with Different Overlapping Rate [3 Clusters]",
        "Overlapping Rate Among Clusters",
        "Rounds",
        "save/over-rounds-his.png",
        0.05,
    )
    avg_rounds.insert(0, 99)
    tts_rounds.insert(0, 89)
    labels = np.insert(labels, 0, 0.1)
    plot_cluster_info(
        avg_rounds,
        tts_rounds,
        labels,
        "Communication Rounds with Different Overlapping Rate [3 Clusters]",
        "Overlapping Rate Among Clusters",
        "Rounds",
        "save/over-rounds-his-with0.1.png",
        0.025,
    )


if __name__ == "__main__":
    # his_cluster()
    his_overlapping()
