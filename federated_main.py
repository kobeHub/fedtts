import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import yaml
import pathlib

import torch
from torch import nn
from tensorboardX import SummaryWriter

from lib.options import args_parser
from lib.round_cache import RoundCacheManager
from lib.update import LocalUpdate, test_inference
from lib.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from lib.utils import get_dataset, average_weights, exp_details, set_seed


if __name__ == "__main__":
    start_time = time.time()

    args = args_parser()
    exp_details(args)
    set_seed(args.seed)

    # define paths
    path_project = os.path.abspath(".")
    logger = SummaryWriter(os.path.join("./logs", f"{args.local_algo}_{args.eid}"))
    save_dirs = [
        os.path.join(os.path.join("save", f"{args.local_algo}_{args.eid}"), i)
        for i in ["pickle", "train_loss_img", "train_acc_img"]
    ]
    for dir in save_dirs:
        if not os.path.exists(dir):
            print(f"Create dir: {dir}")
            pathlib.Path(dir).mkdir(parents=True)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = "cuda" if args.gpu else "cpu"

    # load dataset and user groups
    cache_manager = RoundCacheManager()
    with open(args.config_file, "r") as conf_yaml:
        cluster_config = yaml.safe_load(conf_yaml)
    train_dataset, test_dataset, user_groups = get_dataset(
        args, cache_manager, cluster_config
    )
    cache_manager.eval()

    # BUILD MODEL
    global_model = nn.Module()
    if args.model == "cnn":
        # Convolutional neural netork
        if args.dataset == "mnist":
            global_model = CNNMnist(args=args)
        elif args.dataset == "fmnist":
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == "cifar":
            global_model = CNNCifar(args=args)

    elif args.model == "mlp":
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes)
    else:
        exit("Error: unrecognized model")

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()
    local_algo = args.local_algo

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    end_epoch = 0
    for epoch in tqdm(range(args.epochs)):
        # init local weights and loss
        local_weights, local_losses = [], []
        print(f"\n | Global Training Round : {epoch+1} |\n")

        global_model.train()
        # sample a fraction of users (with args frac)
        m = max(int(args.frac * args.num_users), 1)
        selected_uids = np.random.choice(range(args.num_users), m, replace=False)
        # print(f"Selected users: {selected_uids}, user idx: {user_groups.keys()}")

        # for each sampled user
        for uid in selected_uids:
            local_model = LocalUpdate(
                args=args,
                client_id=uid,
                dataset=train_dataset,
                idxs=user_groups[uid],
                logger=logger,
            )
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                local_algo=local_algo,
                round_cache=cache_manager,
            )
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)
        cache_manager.advance_round()

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(
                args=args,
                client_id=c,
                dataset=train_dataset,
                idxs=user_groups[c],
                logger=logger,
            )
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc) / len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f" \nAvg Training Stats after {epoch+1} global rounds:")
            print(f"Training Loss : {np.mean(np.array(train_loss))}")
            print("Train Accuracy: {:.2f}% \n".format(100 * train_accuracy[-1]))

        end_epoch = epoch
        cur_accuracy = train_accuracy[-1]
        # Check if it reachs a target accuracy.
        if args.target_accuracy != -1 and epoch > args.eval_after:
            if (epoch + 1) % args.eval_every == 0:
                cur_test_acc, _ = test_inference(args, global_model, test_dataset)
                if cur_test_acc >= args.target_accuracy:
                    print(
                        f"| Global rounds: {end_epoch} ",
                        "| test-set accuracy reachs",
                        f": {cur_test_acc} >= {args.target_accuracy}",
                    )
                    break
                else:
                    print(
                        f"| Global rounds: {end_epoch} ",
                        "| test-set accuracy",
                        f": {cur_test_acc} < {args.target_accuracy}",
                    )

    # Test inference after completion of training
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f" \n Results after {end_epoch} global rounds of training:")
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_base_name = (
        f"{args.local_algo}_{args.dataset}"
        f"_{args.model}_R{args.epochs}_C[{args.frac}]_"
        f"iid[{args.iid}]_TA[{args.target_accuracy}]_B[{args.local_bs}]"
        f"_Cluster[{args.n_cluster}]_Over[{args.r_overlapping}]"
        f"_Gamma[{args.gamma}]_Trans[{args.n_transfer}]"
    )
    dump_file = os.path.join(save_dirs[0], file_base_name + ".pkl")
    print(f"Dump training metrics to {dump_file}")

    with open(dump_file, "wb") as f:
        pickle.dump([train_loss, train_accuracy], f)
        pickle.dump([test_acc, test_loss], f)

    print("\n Total Run Time: {0:0.4f}".format(time.time() - start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    # Plot Loss curve
    plt.figure()
    plt.title("Training Loss vs Communication rounds")
    plt.plot(range(len(train_loss)), train_loss, color="r")
    plt.ylabel("Training loss")
    plt.xlabel("Communication Rounds")
    plt.savefig(os.path.join(save_dirs[1], file_base_name + "_loss.png"))

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title("Average Accuracy vs Communication rounds")
    plt.plot(range(len(train_accuracy)), train_accuracy, color="k")
    plt.ylabel("Average Accuracy")
    plt.xlabel("Communication Rounds")
    plt.savefig(os.path.join(save_dirs[2], file_base_name + "_acc.png"))
    logger.close()
