import unittest
from lib.clustering import cluster_dataset
from lib.round_cache import RoundCacheManager
import numpy as np
import yaml
import torchvision


class TestClustering(unittest.TestCase):
    def test_cluster_method(self):
        np.random.seed(0)
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
        overlapping_cnt = 0
        for uid, dp in user_datapoints.items():
            print(uid, dp.shape)
            labels = set(idx_labels[1, np.isin(idx_labels[0, :], dp)])
            print(f"labels: {labels}")
            if len(labels) > 4:
                overlapping_cnt += 1
        assert 6 == overlapping_cnt


if __name__ == "__main__":
    unittest.main()
