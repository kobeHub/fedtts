# Federated Transferrable Training Schema (FedTTS) & FedAvg

Implementation of Federated Transferrable Training Schema and the vanilla federated learning paper : [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).


Experiments are produced on MNIST, Fashion MNIST and CIFAR10 (both IID and non-IID). In case of non-IID, the data amongst the users can be split equally or unequally.

Since the purpose of these experiments are to illustrate the effectiveness of the federated learning paradigm, only simple models such as MLP and CNN are used.

## Requirments

Install all the packages from requirments.txt
* Python3
* Pytorch
* Torchvision

## Data
* Download train and test datasets manually or they will be automatically downloaded from torchvision datasets.
* Experiments are run on Mnist, Fashion Mnist and Cifar.
* To use your own dataset: Move your dataset to data directory and write a wrapper on pytorch dataset class.

## Running the experiments
The baseline experiment trains the model in the conventional way.

* To run the FedAvg with CNN model:
```
python federated_main.py \
				--model=cnn \
        --dataset=mnist \
        --local_ep=5 \
        --epochs=800 \
        --target_accuracy=0.98 \
        --eval_every=2 \
        --local_bs=10 \
        --frac=0.1 \
        --eval_after=300 \
        --local_algo=FedAvg \
        --n_cluster=5 \
        --r_overlapping=0 \
        --gamma=0.5 \
        --n_transfer=5 \
        --config_file=./config/fedtts-conf.yaml \
        --eid=1 \
        --verbose=0
```
* To run the FedTTS with CNN model:
```
python federated_main.py \
				--model=cnn \
        --dataset=mnist \
        --local_ep=5 \
        --epochs=800 \
        --target_accuracy=0.98 \
        --eval_every=2 \
        --local_bs=10 \
        --frac=0.1 \
        --eval_after=100 \
        --local_algo=FedTTS \
        --n_cluster=4 \
        --r_overlapping=0 \
        --gamma=0.5 \
        --n_transfer=5 \
        --config_file=./config/fedtts-conf.yaml \
        --eid=2 \
        --verbose=0
```
You can change the default values of other parameters to simulate different conditions.


## Further Readings
### Papers:
* [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
* [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
* [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)

### Blog Posts:
* [CMU MLD Blog Post: Federated Learning: Challenges, Methods, and Future Directions](https://blog.ml.cmu.edu/2019/11/12/federated-learning-challenges-methods-and-future-directions/)
* [Leaf: A Benchmark for Federated Settings (CMU)](https://leaf.cmu.edu/)
* [TensorFlow Federated](https://www.tensorflow.org/federated)
* [Google AI Blog Post](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
