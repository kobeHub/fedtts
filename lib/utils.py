#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from .clustering import cluster_dataset
from .sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from .sampling import cifar_iid, cifar_noniid


def get_dataset(args, cache_manager, cluster_conf):
    """Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == "cifar":
        data_dir = "../data/cifar/"
        apply_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_dataset = datasets.CIFAR10(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.CIFAR10(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == "mnist" or "fmnist":
        if args.dataset == "mnist":
            data_dir = "../data/mnist/"
            conf = cluster_conf["MNIST"]
        else:
            data_dir = "../data/fmnist/"
            conf = cluster_conf["FMNIST"]

        apply_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=apply_transform
        )

        test_dataset = datasets.MNIST(
            data_dir, train=False, download=True, transform=apply_transform
        )

        # Determine is there are clusters
        if args.n_clusters > 0:
            user_groups = cluster_dataset(
                train_dataset,
                args.n_users,
                args.n_cluster,
                args.overlapping_rate,
                conf,
                cache_manager,
            )
        else:
            # sample training data amongst users
            if args.iid:
                # Sample IID user data from Mnist
                user_groups = mnist_iid(train_dataset, args.num_users)
            else:
                # Sample Non-IID user data from Mnist
                if args.unequal:
                    # Chose uneuqal splits for every user
                    user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                else:
                    # Chose euqal splits for every user
                    user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def compute_local_init(w_list, dp_list, gamma, global_model):
    """
    Compute the initial model for each user by transfer combining
    Other models from different clusters.
    """
    # weighted avg model from other clusters.
    total_dps = float(sum(dp_list))
    transfer_avg = (dp_list[0] / total_dps) * copy.deepcopy(w_list[0])
    for key in transfer_avg.keys():
        for i in range(1, len(w_list)):
            transfer_avg[key] += (dp_list[i] / total_dps) * w_list[i][key]

    # Transfer averaging gamma part from other clusters.
    transfer_avg = average_weights(w_list)
    for key in transfer_avg.keys():
        transfer_avg[key] = transfer_avg[key] * gamma + global_model * (1 - gamma)
    return transfer_avg


def exp_details(args):
    print("\nExperimental details:")
    print(f"    Model     : {args.model}")
    print(f"    Optimizer : {args.optimizer}")
    print(f"    Learning  : {args.lr}")
    print(f"    Global Rounds   : {args.epochs}\n")

    print("    Federated parameters:")
    if args.iid:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args.frac}")
    print(f"    Local Batch size   : {args.local_bs}")
    print(f"    Local Epochs       : {args.local_ep}\n")
    return
