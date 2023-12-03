import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def get_pickles(pickle_dir):
    pickles = os.listdir(pickle_dir)
    train_p, test_p = "", ""
    for p in pickles:
        if "train" in p:
            train_p = p
        else:
            test_p = p
    return os.path.join(pickle_dir, train_p), os.path.join(pickle_dir, test_p)


def plot_traing_results(file_dir):
    pickle_dir = os.path.join(file_dir, "pickle")
    train_p, test_p = get_pickles(pickle_dir)
    print(f"Pickles: {train_p}")
    with open(train_p, "rb") as file:
        data = np.array(pickle.load(file))
    with open(test_p, "rb") as file:
        test_data = np.array(pickle.load(file))
    train_steps, training_loss, training_acc = data[0, :], data[1, :], data[2, :]
    test_steps, test_acc = test_data[0, :], test_data[1, :]
    fig, ax1 = plt.subplots()

    # Plot the loss on the left y-axis
    color = "tab:red"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss", color=color)
    line1 = ax1.plot(train_steps, training_loss, label="Loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    # Create a secondary y-axis to plot accuracy
    ax2 = ax1.twinx()
    color = "tab:blue"
    c3 = "tab:green"
    ax2.set_ylabel("Accuracy", color=color)
    line2 = ax2.plot(train_steps, training_acc, label="Training Accuracy", color=color)
    line3 = ax2.plot(test_steps, test_acc, label="Test Accuracy", color=c3, linewidth=3)
    ax2.tick_params(axis="y", labelcolor=color)

    line3_x_values = [
        test_steps[0],
        test_steps[-1],
    ]  # Use the x-values of the beginning and ending points of line1
    parallel_lines_x_values = np.linspace(
        0, 1, num=100
    )  # Adjust the number of points as needed
    for x in line3_x_values:
        ax1.plot(
            [x] * len(parallel_lines_x_values),
            parallel_lines_x_values,
            color="gray",
            alpha=0.5,
            linestyle="--",
        )

    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="lower right")

    # Show the plot
    exper_name = file_dir.split("/")[1].split("_")[0]
    ta, cluster, over = extract_from_filename(train_p)
    plt.title(
        f"{exper_name} Target Acc: {ta}, Clusters: {cluster}, Overlapping: {over}"
    )
    save_file = os.path.join(file_dir, "training_test.png")
    print(f"Save train test figure {save_file}")
    plt.savefig(save_file, dpi=300)
    plt.close()


def extract_from_filename(input_string):
    # Define regular expressions for extracting values
    ta_pattern = re.compile(r"TA\[(\d+\.\d+)\]")
    cluster_pattern = re.compile(r"Cluster\[(\d+)\]")
    over_pattern = re.compile(r"Over\[(\d+\.\d+)\]")

    # Search for matches in the input string
    ta_match = ta_pattern.search(input_string)
    cluster_match = cluster_pattern.search(input_string)
    over_match = over_pattern.search(input_string)

    # Extract values from matches or set default values
    ta_value = float(ta_match.group(1)) if ta_match else None
    cluster_value = int(cluster_match.group(1)) if cluster_match else None
    over_value = float(over_match.group(1)) if over_match else None

    # Create a dictionary with extracted values
    return ta_value, cluster_value, over_value


def generate_train_figs(save_path):
    experiments = []
    for dir in os.listdir(save_path):
        if os.path.isdir(os.path.join(save_path, dir)):
            experiments.append(os.path.join(save_path, dir))
    print(f"All experiments {experiments}")
    for p in experiments:
        plot_traing_results(p)


def get_train_step_acc(file_dir):
    pickle_dir = os.path.join(file_dir, "pickle")
    train_p, test_p = get_pickles(pickle_dir)
    print(f"Pickles: {train_p}")
    with open(train_p, "rb") as file:
        data = np.array(pickle.load(file))
    return data[0, :], data[2, :]


def plot_cluster(avg_acc, tts_acc, cluster, output):
    plt.plot(
        avg_acc[0],
        avg_acc[1],
        label=f"FedAvg Accuracy - C[{cluster}]",
        color="gray",
        linestyle="-",
    )
    plt.plot(
        tts_acc[0],
        tts_acc[1],
        label=f"FedTTS Accuracy - C[{cluster}]",
        color="cyan",
        linestyle="-",
    )

    # Set plot title
    plt.title(f"FedAvg vs FedTTS Accuracy with {cluster} clusters")

    # Set axis labels
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    # Show legend
    plt.legend()

    print(f"Save fig: {output}")
    # Show the plot
    plt.savefig(output, dpi=300)
    plt.close()


def plot_same_cluster(acc_list, cluster_list, name, line_colors):
    for cluster, acc_tuple in zip(cluster_list, acc_list):
        (steps, accs) = acc_tuple
        plt.plot(
            steps,
            accs,
            label=f"{name} Accuracy - C[{cluster}]",
            color=line_colors[cluster - 3],
            linestyle="-",
        )
    # Set plot title
    plt.title(f"{name} Accuracy across Different Clusters")

    # Set axis labels
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")

    # Show legend
    plt.legend()

    output = f"save/{name}-clusters.png"
    print(f"Save fig: {output}")
    # Show the plot
    plt.savefig(output, dpi=300)
    plt.close()
    # plt.show()


def gen_cluster_compare(avg_list, tts_list, cluster_list):
    avg_step_acc = [get_train_step_acc(dir) for dir in avg_list]
    tts_step_acc = [get_train_step_acc(dir) for dir in tts_list]
    print(f"{avg_step_acc[0][0].shape} {len(tts_step_acc[0])}")
    # Set up the figure
    line_colors = ["blue", "gray", "cyan"]
    styles = ["-", "--", ":"]
    for avg, tts, c in zip(avg_step_acc, tts_step_acc, cluster_list):
        fig = plt.figure()
        plot_cluster(avg, tts, c, f"save/cluster-compare-{c}.png")

    plot_same_cluster(avg_step_acc, cluster_list, "FedAvg", line_colors)
    plot_same_cluster(tts_step_acc, cluster_list, "FedTTS", line_colors)

    # fig = plt.figure()
    # # Plot lines for each set of data
    # for i, (steps, accs) in enumerate(avg_step_acc):
    #     plt.plot(
    #         steps,
    #         accs,
    #         label=f"FedAvg Accuracy - C[{cluster_list[i]}]",
    #         color=line_colors[i],
    #         linestyle="-",
    #     )
    # for i, (steps, accs) in enumerate(tts_step_acc):
    #     plt.plot(
    #         steps,
    #         accs,
    #         color=line_colors[i],
    #         linestyle=styles[i],
    #         label=f"FedTTS - C[{cluster_list[i]}]",
    #     )

    # # Set plot title
    # plt.title("Multiple Lines Plot")

    # # Set axis labels
    # plt.xlabel("Steps")
    # plt.ylabel("Accuracy")

    # # Show legend
    # plt.legend()

    # Show the plot
    # plt.show()


def gen_cc_fig():
    avg_list = [f"save/FedAvg_{suff}" for suff in [1, 2, 3]]
    tts_list = [f"save/FedTTS_{suff}" for suff in [1, 2, 3]]
    gen_cluster_compare(avg_list, tts_list, [3, 4, 5])


if __name__ == "__main__":
    generate_train_figs("save")
    gen_cc_fig()
