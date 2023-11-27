#!/bin/bash

python federated_main.py --model=cnn \
	--dataset=mnist \
	--local_ep=2 \
	--epochs=5000 \
	--target_accuracy=0.89 \
	--eval_every=2 \
	--local_bs=10 \
	--frac=0.2 \
	--eval_after=30 \
	--local_algo=fedtts \
	--n_cluster=3 \
	--r_overlapping=0.1 \
	--n_transfer=1 \
	--verbose=0 \
	--config_file=./config/fedtts-conf.yaml
