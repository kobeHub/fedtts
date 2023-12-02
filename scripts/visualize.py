import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import re


def plot_traing_results(file_dir):
    pickle_dir = os.path.join(file_dir, "pickle")
    pickles = os.listdir(pickle_dir)
    train_p, test_p = "", ""
    for p in pickles:
        if "train" in p:
            train_p = p
        else:
            test_p = p
    train_p, test_p = os.path.join(pickle_dir, train_p), os.path.join(
        pickle_dir, test_p
    )
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
    exper_name = file_dir.split("/")[1][:-2]
    ta, cluster, over = extract_from_filename(train_p)
    plt.title(
        f"{exper_name} Target Acc: {ta}, Clusters: {cluster}, Overlapping: {over}"
    )
    save_file = os.path.join(file_dir, "training_test.png")
    print(f"Save train test figure {save_file}")
    plt.savefig(save_file, dpi=300)


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


if __name__ == "__main__":
    generate_train_figs("save")

# Now 'loaded_data' contains the content of the pickle file
