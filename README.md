# Federated learning under heterogeneous Systems

This repository is the implementation of Fjord.

Federated Learning is a distributed framework that allows a large number of IoT devices keeping their data locally and collaborating to learn a global machine-learning model by aggregating locally computed updates.  %More precisely, a master coordinates %participating devices and each client owns a local training dataset that is never updated %to the server.  Each client computes an update with respect to the current global model and its own local dataset and communicates only this update with the server. This framework reduces the privacy leakage risks in the centralized scenarios such as cloud computing where the devices’ data is required to be transferred to the server for the future process. The traditional algorithms for FL assume that the involved devices have the same storage and computation capacities. However, due to the variability in hardware (GPU, CPU, RAM) of these federated devices, this assumption is no longer realistic which forces the traditional algorithms to either drop the low-tier devices or limit the global model’s size to accommodate the weakest devices.  Therefore, in large-scale deployments, the heterogeneity of the system constitutes a challenge for fairness and training performance. 

This project's goal is to understand the advanced methods proposed for tackling the above-mentioned issues when learning among varying types of devices. More precisely, we study and implement one algorithm related to FL (FjORD) using PyTorch. We analyze the obtained performance results and to show the beneficial cases when applying FjORD.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Usage

We provide code to simulate federated training of machine learning. 
The core objects are `Aggregator` and `Client`, different federated learning
algorithms can be implemented by revising the local update method 
`Client.step()` and/or the aggregation protocol defined in
`Aggregator.mix()` and `Aggregator.update_client()`.

In addition to the trivial baseline consisting of training models locally without
any collaboration, this repository supports the following federated learning
algorithms:


* FjORD ([Leontiadis et al. 2021](https://arxiv.org/abs/2102.13451))
* FedAvg ([McMahan et al. 2017](http://proceedings.mlr.press/v54/mcmahan17a.html))
* FdProx ([Li et al. 2018](https://arxiv.org/abs/1812.06127))
* Clustered FL ([Sattler et al. 2019](https://ieeexplore.ieee.org/abstract/document/9174890))
* pFedMe ([Dinh et al. 2020](https://proceedings.neurips.cc/paper/2020/file/f4f1f13c8289ac1b1ee0ff176b56fc60-Paper.pdf))
* L2SGD ([Hanzely et al. 2020](https://proceedings.neurips.cc/paper/2020/file/187acf7982f3c169b3075132380986e4-Paper.pdf))
* APFL ([Deng et al. 2020](https://arxiv.org/abs/2003.13461))
* q-FFL ([Li et al. 2020](https://openreview.net/forum?id=ByexElSYDr))
* AFL ([Mohri et al. 2019](http://proceedings.mlr.press/v97/mohri19a.html))

## Datasets

We provide five federated benchmark datasets spanning a wide range
of machine learning tasks: image classification (CIFAR10 and CIFAR100),
handwritten character recognition (EMNIST and FEMNIST), and language
modelling (Shakespeare), in addition to a synthetic dataset

Shakespeare dataset (resp. FEMNIST) was naturally partitioned by assigning
all lines from the same characters (resp. all images from the same writer)
to the same client.  We created federated versions of CIFAR10 and EMNIST by
distributing samples with the same label across the clients according to a 
symmetric Dirichlet distribution with parameter 0.4. For CIFAR100,
we exploited the availability of "coarse" and "fine" labels, using a two-stage
Pachinko allocation method  to assign 600 sample to each of the 100 clients.

The following table summarizes the datasets and models

|Dataset         | Task |  Model |
| ------------------  |  ------|------- |
| FEMNIST   |     Handwritten character recognition       |     2-layer CNN + 2-layer FFN  |
| EMNIST    |    Handwritten character recognition     |      2-layer CNN + 2-layer FFN     |
| CIFAR10   |     Image classification        |      MobileNet-v2 |
| CIFAR100    |     Image classification         |      MobileNet-v2  |
| Shakespeare |     Next character prediction        |      Stacked LSTM    |
| Synthetic dataset| Binary classification | Linear model | 

See the `README.md` files of respective dataset, i.e., `data/$DATASET`,
for instructions on generating data

## Training

Run on one dataset, with a specific  choice of federated learning method.
Specify the name of the dataset (experiment), the used method, and configure all other
hyper-parameters (see all hyper-parameters values in the appendix of the paper)

Fjord on emnist dataset experiment
``` 

 module load conda/2021.11-python3.9 

 python3 run_experiment.py emnist Fjord \
    --n_learners 1 \
    --n_rounds 300 \
    --bz 16\
    --lr 0.1 \
    --lr_scheduler multi_step \
    --log_freq 1 \
    --device cpu \
    --optimizer sgd \
    --seed 12345 \
    --logs_root ./logs_cifar10 \
    --verbose 1\
    --k 5\
    --sampling_rate 0.1
```

FedAvg experiment
```train
python3  python3 run_experiment.py medmnist FedAvg \
    --n_learners 1 \
    --n_rounds 50 \
    --bz 128 \
    --lr 0.01 \
    --lr_scheduler multi_step \
    --log_freq 2 \
    --device cpu \
    --optimizer sgd \
    --seed 1234 \
    --logs_root ./logs \
    --verbose 1
```

The test and training accuracy and loss will be saved in the specified log path.

We provide example scripts to run paper experiments under `scripts/` directory.

## Evaluation

We give instructions to run experiments on CIFAR-10 dataset as an example
(the same holds for the other datasets). You need first to go to 
`./data/cifar10`, follow the instructions in `README.md` to download and partition
the dataset.

All experiments will generate tensorboard log files (`logs/cifar10`) that you can 
interact with, using [TensorBoard](https://www.tensorflow.org/tensorboard)


### Average performance of personalized models

Run the following scripts, this will generate tensorboard logs that you can interact with to make plots or get the
values presented in Table 2

```eval
# run FedAvg
echo "Run FedAvg"
python run_experiment.py cifar10 FedAvg --n_learners 1 --n_rounds 200 --bz 128 --lr 0.01 \
 --lr_scheduler multi_step --log_freq 5 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedAvg + local adaption
echo "run FedAvg + local adaption"
python run_experiment.py cifar10 FedAvg --n_learners 1 --locally_tune_clients --n_rounds 201 --bz 128 \
 --lr 0.001 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run training using local data only
echo "Run Local"
python run_experiment.py cifar10 local --n_learners 1 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run Clustered FL
echo "Run Clustered FL"
python run_experiment.py cifar10 clustered --n_learners 1 --n_rounds 201 --bz 128 --lr 0.003 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1

# run FedProx
echo "Run FedProx"
python run_experiment.py cifar10 FedProx --n_learners 1 --n_rounds 201 --bz 128 --lr 0.01 --mu 1.0\
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# Run pFedME
echo "Run "
python run_experiment.py cifar10 pFedMe --n_learners 1 --n_rounds 201 --bz 128 --lr 0.001 --mu 1.0 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer prox_sgd --seed 1234 --verbose 1

# run FedEM
echo "Run FedEM"
python run_experiment.py cifar10 FedEM --n_learners 3 --n_rounds 201 --bz 128 --lr 0.03 \
 --lr_scheduler multi_step --log_freq 10 --device cuda --optimizer sgd --seed 1234 --verbose 1
```

Similar for other datasets are provided in `papers_experiments/`

### Generalization to unseen clients

You need to run the same script as in the previous section. Make sure that `--test-clients-frac` is non-zero,
when you call `generate_data.py`.

### Clients sampling

Our code gives the possibility to use only a fraction of the available clients at each round,
you can specify this  parameter when running `run_experiment.py` using the argument `--sampling_rate` (default is `0`). 

### Fully-decentralized federated learning

To simulate a fully-decentralized training you need to specify `--decentralized` when you run `run_experiment.py`

## Results

The  performance of each personalized model (which is the same for all clients 
in the case of FedAvg and FedProx) is evaluated on the local test dataset
(unseen at training). The following shows the average weighted accuracy with 
weights proportional to local dataset sizes. We observe that FedEM obtains 
the best performance across all datasets.


|Dataset         | Local | FedAvg| FedAvg+ | FedEM | FjORD (ours) |
| ------------------  |  ------|-------|----------------       | -------------- |--------------|
| FEMNIST   |     71.0       |     78.6              |75.3        | 79.9| 74.081
| EMNIST    |    71.9     |      82.6              |83.1          |83.5| 81
| CIFAR10   |     70.2        |      78.2             |82.3          |84.3|
| CIFAR100    |     31.5        |      40.9              |39.0         |44.1|
| Shakespeare |     32.0        |      46.7              |    40.0      |43.7|



We can also visualise the evolution of the train loss, train accuracy, test loss and test accuracy for CIFAR-10 dataset

![](https://user-images.githubusercontent.com/42912620/120851258-de39a800-c578-11eb-8d0e-13460e5d71cc.PNG)

Similar plots can be built for other experiments using the `make_plot` function in `utils/plots.py`