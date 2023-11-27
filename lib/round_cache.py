import numpy as np
import unittest


class RoundCacheManager:
    """
    A Singleton class to manage the device's paramters in each round.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoundCacheManager, cls).__new__(cls)
            # client_id -> models in last round
            cls._instance.client_to_models_last_round = dict()
            # client_id -> models in current round
            cls._instance.client_to_models_current_round = dict()
            # client_id -> [clsuter_id]
            cls._instance.client_to_clusters = dict()
            # cluster_id -> [client_id]
            cls._instance.cluster_to_clients = dict()
            # cluster_id -> int; the actual user counts in each cluster.
            cls._instance.cluster_user_cnt_list = list()
            # client_id -> int; the datapoints counts in each user.
            cls._instance.client_to_datapoint_cnt = dict()
            cls._instance.cache_rounds = 0
        return cls._instance

    def advance_round(self):
        """
        Set the round number of cached models.
        """
        self.client_to_models_last_round.clear()
        self.client_to_models_last_round.update(self.client_to_models_current_round)
        self.client_to_models_current_round.clear()
        self.cache_rounds += 1

    def get_round(self):
        return self.cache_rounds

    def add_model(self, client_id, model):
        self.client_to_models_current_round[client_id] = model

    def add_client_to_cluster(self, client_id, cluster_id):
        if client_id not in self.client_to_clusters:
            self.client_to_clusters[client_id] = []
        self.client_to_clusters[client_id].append(cluster_id)
        if cluster_id not in self.cluster_to_clients:
            self.cluster_to_clients[cluster_id] = []
        self.cluster_to_clients[cluster_id].append(client_id)

    def sample_from_other_cluster(self, client_id, size_per_cluster):
        """
        Sample models from last round in other clusters.
        :return List of (client_id, client_model, client_datapoint_cnt)
        """
        client_ids, models, dp_cnts = [], [], []
        clusters = set(self.cluster_to_clients.keys()) - set(
            self.client_to_clusters[client_id]
        )
        # print(f"Avail cluster: {clusters}")
        for avail_cluster in clusters:
            clients = self.cluster_to_clients[avail_cluster]
            # print(f"cluster: {avail_cluster}, client: {clients}")
            # Filter the cache clients.
            clients = [
                client
                for client in clients
                if client in self.client_to_models_last_round
            ]
            if len(clients) < size_per_cluster:
                continue
            for cli_id in np.random.choice(
                clients, size=size_per_cluster, replace=False
            ):
                if cli_id != client_id:
                    client_ids.append(cli_id)
                    models.append(self.client_to_models_last_round[cli_id])
                    dp_cnts.append(self.client_to_datapoint_cnt[cli_id])
        return client_ids, models, dp_cnts

    def set_cluster_user_cnt(self, user_cnt_list):
        self.cluster_user_cnt_list = user_cnt_list

    def set_client_datapoint_cnt(self, client_id, dp_cnt):
        self.client_to_datapoint_cnt[client_id] = dp_cnt

    def eval(self):
        print(f"===========Eval Round Cache for round: {self.cache_rounds}============")
        print(f"Cluster set: {set(self.cluster_to_clients.keys())}")
        print(f"Clients set: {set(self.client_to_clusters.keys())}\n")
        for cluster, clients in self.cluster_to_clients.items():
            print(
                f"Cluster {cluster} has clients (Actual cnts: "
                f"{self.cluster_user_cnt_list[cluster]}; "
                f"logical cnts: {len(clients)}) : {clients}"
            )
        print(
            f"\nCached models from clients in last round: {self.client_to_models_last_round.keys()}"
        )
        print(
            f"\nCached models from clients in current round: {self.client_to_models_current_round.keys()}"
        )
        print(f"Device datapoint glances:\n{self.client_to_datapoint_cnt}")
        print(
            f"======================================================================="
        )
