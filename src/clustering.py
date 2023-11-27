import numpy as np
import unittest
import yaml
import torchvision

from round_cache import RoundCacheManager


def cluster_dataset(
    dataset,
    n_users,
    n_cluster,
    overlapping_rate,
    mnist_conf,
    round_cache: RoundCacheManager,
):
    """
    Partition dataset into `n_cluster` and each cluster is highly non-IID
    :param dataset: input dataset
    :param n_users: number of total users
    :param n_cluster: number of clusters
    :param overlapping_rate: input dataset
    :return: dict of image index
    """
    if n_cluster not in mnist_conf["n_clusters"]:
        raise ValueError(f"Give n_cluster={n_cluster} is not defined in config file")
    if n_cluster not in mnist_conf["avail_overlapping"]:
        raise ValueError(f"Give n_cluster={n_cluster} is not allowed with overlapping")

    labels_cnt_list = mnist_conf["labels_per_cluster"][n_cluster]
    overlapping_topo = mnist_conf["overlapping"][n_cluster]

    # The number of data points per user
    data_size = len(dataset)

    # Sort idxs according to labels.
    idxs = np.arange(data_size)
    labels = dataset.targets.numpy()
    idxs_labels = np.vstack((idxs, labels))

    # Assign labels to each cluster;
    labels_set = list(set(labels))
    np.random.shuffle(labels_set)
    label_set_idx = compute_ranges_with_cnt(labels_cnt_list, len(labels_set))
    cluster_labels = [
        labels_set[label_set_idx[i - 1] : label_set_idx[i]]
        for i in range(1, len(label_set_idx))
    ]
    cluster_data_idxs = divide_cluster_idx(idxs_labels, cluster_labels)

    # Split cluster according to label counts; [30, 30, 40]
    user_cnt_per_label = int(n_users / len(labels_set))
    user_cnts = [(user_cnt_per_label * lc) for lc in labels_cnt_list[:-1]]
    user_cnts.append(n_users - sum(user_cnts))
    user_ranges = compute_ranges_with_cnt(user_cnts, n_users)
    cluster_user_idxs = [
        np.arange(user_ranges[i - 1], user_ranges[i])
        for i in range(1, len(user_ranges))
    ]
    # Assign clients to clusters.
    round_cache.set_cluster_user_cnt(user_cnts)
    for cid, uid_list in enumerate(cluster_user_idxs):
        for uid in uid_list:
            round_cache.add_client_to_cluster(uid, cid)

    # Divide datapoint to each client.
    user_datapoints = {}
    for cluster_idx, user_idx in zip(cluster_data_idxs, cluster_user_idxs):
        user_datapoints.update(divide_within_cluster(cluster_idx, user_idx))

    # Arrange overlapping
    if overlapping_rate > 0.0 and overlapping_rate < 1.0:
        for lapping_part in overlapping_topo:
            lhs, rhs = [int(li) for li in lapping_part.split("&")]
            lhs_over_cnt, rhs_over_cnt = int(user_cnts[lhs] * overlapping_rate), int(
                user_cnts[rhs] * overlapping_rate
            )
            # Select overlapping users randomly.
            lhs_user_idxs = np.random.choice(
                cluster_user_idxs[lhs], size=lhs_over_cnt, replace=False
            )
            rhs_user_idxs = np.random.choice(
                cluster_user_idxs[rhs], size=rhs_over_cnt, replace=False
            )
            overlapping_dict = mix_and_divide(
                user_datapoints, lhs_user_idxs, rhs_user_idxs
            )
            # Update the overlapping dict and cache manager.
            user_datapoints.update(overlapping_dict)
            for uid in lhs_user_idxs:
                round_cache.add_client_to_cluster(uid, rhs)
            for uid in rhs_user_idxs:
                round_cache.add_client_to_cluster(uid, lhs)
    # Record datapoint count for each client.
    for uid, datapoints in user_datapoints:
        round_cache.set_client_datapoint_cnt(uid, datapoints.shape[0])

    return user_datapoints, idxs_labels


def divide_cluster_idx(idx_labels, cluster_labels):
    """
    Devide the data into clusters according to the labels.
    :return List of the divided cluster datapoints indice [np.array([0, 2, ..]), ...]
    """
    labels = idx_labels[1, :]
    cluster_idxs = []
    for clabels in cluster_labels:
        mask = np.isin(labels, clabels)
        cluster_idxs.append(idx_labels[0, mask])
    return cluster_idxs


def divide_within_cluster(cluster_idxs: np.ndarray, user_idxs: np.ndarray):
    """
    Divide the cluster data points to users belongs to this cluster equally.
    :return Dict of user's datapoints.
    """
    user_cnt = user_idxs.shape[0]
    datapoints_cnt = cluster_idxs.shape[0]
    per_user_cnt = int(datapoints_cnt / (user_cnt + 1))
    dict_users = {}
    np.random.shuffle(cluster_idxs)
    left, right = 0, per_user_cnt
    for uid in user_idxs[:-1]:
        dict_users[uid] = cluster_idxs[left:right]
        left = right
        right += per_user_cnt
    dict_users[user_idxs[-1]] = cluster_idxs[right:]
    return dict_users


def mix_and_divide(
    user_datapoints, lhs_user_idxs: np.ndarray, rhs_user_idxs: np.ndarray
):
    """
    Mix the datapoints of the 2 collections, shuffle and then divide
    to each clients.

    :return a dict containing the overlapping user datapoints.
    """
    # Merge data points for users in lhs_user_idxs and rhs_user_idxs
    merged_data = np.concatenate(
        [
            user_datapoints.get(user_idx, np.array([]))
            for user_idx in np.concatenate([lhs_user_idxs, rhs_user_idxs])
        ]
    )

    # Shuffle the merged data randomly
    np.random.shuffle(merged_data)

    # Divide the shuffled data into batches of the same size for each user
    divided_data = {}
    current_index = 0

    for user_idx in np.concatenate([lhs_user_idxs, rhs_user_idxs]):
        user_data_size = len(user_datapoints.get(user_idx, np.array([])))
        end_index = current_index + user_data_size

        divided_data[user_idx] = merged_data[current_index:end_index]
        current_index = end_index

    return divided_data


def compute_ranges_with_cnt(cnt_list, total_length):
    """
    Return range list with given elements cnt: [3, 3, 4] -> [0, 3, 6, 10]
    """
    assert sum(cnt_list) == total_length
    ranges = [0] + cnt_list
    for i in range(1, len(ranges)):
        ranges[i] += ranges[i - 1]
    return ranges


class TestClustering(unittest.TestCase):
    def test_clustering(self):
        n_users, n_cluster, tau = 100, 3, 0.1
        conf_file = "./config/fedtts-conf.yaml"
        cache_manager = RoundCacheManager()
        with open(conf_file, "r") as conf_yaml:
            cluster_config = yaml.safe_load(conf_yaml)
        dataset = torchvision.datasets.MNIST(root="./data/mnist", download=True)
        user_datapoints, idx_labels = cluster_dataset(
            dataset, n_users, n_cluster, tau, cluster_config["MNIST"], cache_manager
        )
        cache_manager.eval()
        print(len(user_datapoints))
        for uid, dp in user_datapoints.items():
            print(uid, dp.shape)
            labels = set(idx_labels[1, np.isin(idx_labels[0, :], dp)])
            print(f"labels: {labels}")


unittest.main()
