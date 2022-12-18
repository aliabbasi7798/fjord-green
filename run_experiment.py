"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""
from utils.utils import *
from utils.constants import *
from utils.args import *

from tensorboardX import SummaryWriter


def init_clients(args_, root_path, logs_root):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators =\
        get_loaders(
            type_=LOADER_TYPE[args_.experiment],
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation
        )

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue
# get client model instance 
        learners_ensemble =\
            get_learners_ensemble(
                n_learners=args_.n_learners,# we have 1 learner
                name=args_.experiment,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=args_.input_dimension,
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu
            )

        logs_path = os.path.join(logs_root, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients,
            k=args_.k,
            green = random.randint(0,1)
        )
        # here we send value k to the client, and a function attributes a random maximum capability, based on this
        # max_cap the server send a F_max subnetwork
        client.green_compute()

        clients_.append(client)

    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train")
    )
    for c in clients:
        print(c.k)
    print("==> Test Clients initialization..")
    test_clients = init_clients(
        args_,
        root_path=os.path.join(data_dir, "test"),
        logs_root=os.path.join(logs_root, "test")
    )

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu
        )

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed,
            k=args.k
        )
    tr_acc, tr_round = [], []
    print("Training..")
    # just a progress bar
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:
        tr_1, tr_2 = aggregator.mix()
        if(len(tr_1) > 0):
            tr_acc.append(tr_1[0])
            tr_round.append(tr_2[0])
        #print(1)
        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round
        #print(2)
    if "save_path" in args_:
        save_root = os.path.join(args_.save_path)
        #print(3)
        os.makedirs(save_root, exist_ok=True)
        aggregator.save_state(save_root)
    return tr_acc, tr_round

if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    tr_acc, tr_round = run_experiment(args)

    print(tr_acc)
    print(tr_round)

    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    # Plot Loss curve
    '''
    plt.figure()
    plt.title('Train accuracy')
    # Plot Loss curve
    plt.plot(tr_round, tr_acc, color='r', label='Fjord')

    plt.legend()
    plt.ylabel('test accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('save/plot3.png')
    # importing the csv module
    '''
    import csv
    k = 0
    # field names
    fields = ['Train Accuracy', 'Rounds' , 'k']
    rows = []
    # data rows of csv file
    for i in range(len(tr_round)):
        rows.append([tr_round[i] , tr_acc[i] , k])

    # name of csv file
    filename = "do(k=green_1).csv"

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(rows)

