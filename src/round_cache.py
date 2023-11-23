from numpy import random
import unittest


class RoundCacheManager:
    """
    A Singleton class to manage the device's paramters in each round.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoundCacheManager, cls).__new__(cls)
            # client_id -> models
            cls._instance.client_to_models = dict()
            # client_id -> [clsuter_id]
            cls._instance.client_to_clusters = dict()
            # cluster_id -> [client_id]
            cls._instance.cluster_to_clients = dict()
            cls._instance.cache_rounds = 0
        return cls._instance

    def set_round(self, round):
        """
        Set the round number of cached models.
        """
        if round > self.cache_rounds:
            self.client_to_models.clear()
        self.cache_rounds = round

    def get_round(self):
        return self.cache_rounds

    def add_model(self, client_id, model):
        self.client_to_models[client_id] = model

    def add_client_to_cluster(self, client_id, cluster_id):
        if client_id not in self.client_to_clusters:
            self.client_to_clusters[client_id] = []
        self.client_to_clusters[client_id].append(cluster_id)
        if cluster_id not in self.cluster_to_clients:
            self.cluster_to_clients[cluster_id] = []
        self.cluster_to_clients[cluster_id].append(client_id)

    def record_model(self, client_id, model):
        self.client_to_models[client_id] = model

    def sample_from_other_cluster(self, client_id):
        samples = []
        clusters = set(self.cluster_to_clients.keys()) - set(
            self.client_to_clusters[client_id]
        )
        for avail_cluster in clusters:
            clients = self.cluster_to_clients[avail_cluster]
            cli_id = random.choice(clients)
            if cli_id != client_id:
                samples.append((cli_id, self.client_to_models[cli_id]))
        return samples

    def eval(self):
        print(f"===========Eval Round Cache for round: {self.cache_rounds}============")
        print(f"Cluster set: {set(self.cluster_to_clients.keys())}")
        print(f"Clients set: {set(self.client_to_clusters.keys())}\n")
        for cluster, clients in self.cluster_to_clients.items():
            print(f"Cluster {cluster} has clients: {clients}")
        print(f"\nCached models from clients: {self.client_to_models.keys()}")
        print(
            f"======================================================================="
        )


class TestCache(unittest.TestCase):
    def cacheUsage(self):
        # Creating instances
        singleton_instance1 = RoundCacheManager()
        singleton_instance2 = RoundCacheManager()

        singleton_instance1.set_round(121)
        cc = 0
        for i in range(5):
            for j in range(10):
                client = random.randint(0, 100)
                cc = client
                singleton_instance1.add_client_to_cluster(client, i)
                singleton_instance2.add_model(
                    client, {"client": client, "i": i, "j": j}
                )
        singleton_instance1.eval()
        print(f"cc: {cc}")
        print(singleton_instance2.sample_from_other_cluster(cc))
