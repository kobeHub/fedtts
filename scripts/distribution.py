import matplotlib.pyplot as plt
import numpy as np
from lib.clustering import cluster_dataset
from lib.round_cache import RoundCacheManager
from lib.sampling import mnist_iid
from torchvision import datasets
import numpy as np
import yaml


def plot_circle(x, y, m):
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = x + m * np.cos(theta)
    circle_y = y + m * np.sin(theta)

    plt.plot(circle_x, circle_y, label="Circle")
    plt.scatter(x, y, color="red", label="Center Point")  # Mark the center point
    plt.title(f"Circle with Radius {m} at ({x}, {y})")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")  # Ensure equal scaling on both axes
    plt.show()


def plot_datapoints(datapoint_cnts, factor, name, save):
    plt.figure(figsize=(15, 10))
    for uid, cnts in datapoint_cnts.items():
        x = uid
        for y, m in cnts.items():
            # theta = np.linspace(0, 2 * np.pi, 100)
            # circle_x = x + m / factor * np.cos(theta)
            # circle_y = y + m / factor * np.sin(theta)

            # plt.plot(circle_x, circle_y, color="orange", fill=True)
            circle = plt.Circle((x, y), m / factor, color="orange", fill=True)
            plt.gca().add_patch(circle)
            plt.ylim(-1, 10)
            plt.xlim(0, 101)
    plt.title(f"Global Data Distribution Across Clients-{name}")
    plt.xlabel("Client ID")
    plt.ylabel("Class Labels")
    plt.xticks(np.arange(0, 100, 5))
    plt.yticks(np.arange(0, 10, 1))
    # plt.legend()
    # plt.grid(True)
    # plt.axis("equal")  # Ensure equal scaling on both axes
    # plt.show()
    print(f"Save file {save}")
    plt.savefig(save, dpi=300)
    plt.close()


def demo():
    x_center = 2
    y_center = 3
    radius = 4
    plot_circle(x_center, y_center, radius)


def plot_dist(iid, n_cluster, over):
    dataset = datasets.MNIST(root="./data/mnist")
    n_user = 100
    labels = dataset.targets.numpy()
    # idxs_labels = np.vstack((np.arange(), labels))
    if iid:
        user_dps = mnist_iid(dataset, n_user)
        name = "IID"
        save_name = "save/iid-dis.png"
    else:
        cache = RoundCacheManager()
        with open("./config/fedtts-conf.yaml", "r") as conf_yaml:
            cluster_config = yaml.safe_load(conf_yaml)
        user_dps, _ = cluster_dataset(
            dataset,
            n_user,
            n_cluster,
            over,
            cluster_config["MNIST"],
            cache,
        )
        name = f"Cluster_{n_cluster}_Over_{over}"
        save_name = f"save/cluster_{n_cluster}_over_{over}.png"
    # print(user_dps.keys())
    points_dict = {}
    for uid, data in user_dps.items():
        cnts_dict = {}
        for dp in data:
            label = labels[dp]
            if label not in cnts_dict:
                cnts_dict[label] = 0
            cnts_dict[label] += 1
        points_dict[uid] = cnts_dict
    # print(points_dict)

    plot_datapoints(points_dict, 600, name, save_name)


if __name__ == "__main__":
    # demo()
    plot_dist(False, 3, 0)
    plot_dist(False, 4, 0)
    plot_dist(False, 5, 0)
    plot_dist(False, 3, 0.1)
    plot_dist(False, 3, 0.2)
    plot_dist(False, 3, 0.5)
    plot_dist(False, 3, 0.8)
    plot_dist(False, 5, 0.8)
