import unittest
from lib.round_cache import RoundCacheManager
import numpy as np


class TestCache(unittest.TestCase):
    def test_cache_usage(self):
        # Creating instances
        singleton_instance1 = RoundCacheManager()
        singleton_instance2 = RoundCacheManager()
        np.random.seed(0)

        client_set = np.random.choice(np.arange(100), size=50, replace=False).reshape(
            5, 10
        )
        cc = 0
        for i in range(5):
            for j in range(10):
                client = client_set[i, j]
                cc = client
                singleton_instance1.add_client_to_cluster(client, i)
                singleton_instance2.add_model(
                    client, {"client": client, "i": i, "j": j}
                )
                singleton_instance1.set_client_datapoint_cnt(client, i * 10 + j)
        singleton_instance1.set_cluster_user_cnt([10 for _ in range(5)])
        assert len(singleton_instance1.client_to_models_last_round) == 0
        assert len(singleton_instance1.client_to_models_current_round) == len(
            singleton_instance1.client_to_clusters.keys()
        )
        singleton_instance1.eval()
        print(f"cc: {cc}")
        print(singleton_instance2.sample_from_other_cluster(cc, 1))
        singleton_instance1.advance_round()
        assert len(singleton_instance1.client_to_models_current_round) == 0
        assert len(singleton_instance1.client_to_models_last_round) == len(
            singleton_instance1.client_to_clusters.keys()
        )
        print(singleton_instance2.sample_from_other_cluster(cc, 1))


if __name__ == "__main__":
    unittest.main()
