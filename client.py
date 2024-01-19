import torch.nn.functional as F
import random
import time
from copy import deepcopy
from utils.torch_utils import *
import utils.functions as functions

class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            k,
            green,
            energyClient,
            carbonIntensity,
            tune_locally=False,
            seed=None

    ):
        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally
        self.max_cap, self.possible_p_list = functions.select_max_cap(k=k)  #sample  a max_capability of client and a list of possible dropout rates
        #print("p list" , self.possible_p_list)
        self.green = green
        self.energyClient = energyClient
        self.carbonIntensity = carbonIntensity
        if self.tune_locally:
            self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
        else:
            self.tuned_learners_ensemble = None

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger
        self.__p = self.selectgreen_p()
    def selectgreen_p(self):
        temp = self.carbonIntensity * self.energyClient
        if(self. green == 0):
            if(self.carbonIntensity == 1000 ):
                return 0.2
            elif(self.carbonIntensity == 10 ):
                return 0.6
            else:
                return 1
        elif (self.green == -10):
            if (self.carbonIntensity == 100):
                return 0.6
            elif (self.carbonIntensity == 10):
                return 0.6
            else:
                return 0.6
        elif(self.green == -1):
            if (self.carbonIntensity <= 100):
                return 1
            elif (self.carbonIntensity <= 400):
                return 0.6
            else:
                return 0.2
        elif (self.green == -3):
            if (self.carbonIntensity >= 200):
                return 0.2
            else:
                return 1
        elif(self.green == -5):
            if (self.carbonIntensity == 10 or self.carbonIntensity == 100 or self.carbonIntensity == 1000):
                return 0.5
            else:
                return 0.7
        elif (self.green == -6):
            if (self.carbonIntensity == 100 or self.carbonIntensity == 1000):
                return 0.5
            elif (self.carbonIntensity == 1 or self.carbonIntensity == 10):
                return 0.6
            else:
                return 0.7
        elif (self.green == -2):
            if (self.carbonIntensity > 901 and self.carbonIntensity <= 1124):
               return 0.2
            elif (self.carbonIntensity <= 901 and self.carbonIntensity > 678):
                return 0.4
            elif (self.carbonIntensity > 456 and self.carbonIntensity <=678):
                return 0.6
            elif (self.carbonIntensity <= 456 and self.carbonIntensity > 233):
                return 0.8
            else:
                return 1
        elif (self.green == -4):
            return  round(1.2 - round((self.carbonIntensity-11)/(1113/0.8) + 0.21 , 1) , 1)
        else:
            return self.green
    def select_p(self):
        #print ("green"  , self.possible_p_list)
       # self.__p = self.rng.choice(self.possible_p_list)
        self.__p = self.selectgreen_p()
        #print("p" , self.__p)
    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch


    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform one step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()
        self.select_p()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights,
                    p=self.__p
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights,
                    p=self.__p
                )

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self):
        #print("log1c")
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)

            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)
        #print("log2c")
        #print(train_loss)
        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)
        #print("log3c")

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        if not self.tune_locally:
            return

        for learner_id, learner in enumerate(self.tuned_learners_ensemble):
            copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
            learner.fit_epochs(self.train_iterator, self.local_steps, weights=self.samples_weights[learner_id], p=self.__p)
    def green_compute(self):
        if (self.green == 2):
            self.k= 1
        elif(self.green == 1):
            self.k = 2
        else:
            self.k = 5
    def get_k(self):
        return self.k

class MixtureClient(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)


class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)
