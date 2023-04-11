import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


class MedMnistBase(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class FjordFemnistCNN(nn.Module):
    """
        Implements a model with two convolutional layers followed by pooling, and a final dense layer with 10 units.
    """

    def __init__(self, num_classes):
        super(FjordFemnistCNN, self).__init__()
        self.p = 1
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(980, num_classes)

    def _masked(self, param, prev_param=None, dim=0):
        if dim == 0:
            _param = param[param.mask]
        else:
            _param = param[:, param.mask]
        if prev_param is not None:
            _param = _param[:, prev_param.mask]
        return _param

    def _get_mask(self, layer, dim=0, dropout_rate=1):
        N = layer.shape[dim]
        mask = np.zeros(N).astype(bool)
        mask[:int(N * dropout_rate)] = True
        return mask

    def compute_masks(self):
        self.conv1.weight.mask = self._get_mask(self.conv1.weight, dropout_rate=self.p)
        self.conv1.bias.mask = self._get_mask(self.conv1.bias, dropout_rate=self.p)

        self.conv2.weight.mask = self._get_mask(self.conv2.weight, dropout_rate=self.p)
        self.conv2.weight.prev_mask = self.conv1.weight.mask
        self.conv2.bias.mask = self._get_mask(self.conv2.bias, dropout_rate=self.p)

        self.fc1.weight.mask = self._get_mask(self.fc1.weight, dim=1, dropout_rate=self.p)

    def forward(self, x, p=1):
        self.p = p
        self.compute_masks()
        x = F.conv2d(
            x,
            self._masked(self.conv1.weight),
            self._masked(self.conv1.bias),
            1,  # stride
            2  # padding
        )
        x = self.pool(F.relu(x))
        # Second conv
        x = F.conv2d(
            x,
            self._masked(self.conv2.weight, self.conv1.weight),
            self._masked(self.conv2.bias),
            1,  # stride
            2  # padding
        )
        x = self.pool(F.relu(x))
        x = x.view(x.size(0), -1)
        x = F.linear(x, self._masked(self.fc1.weight, dim=1), self.fc1.bias)
        return x

class FjordCifar10CNN(nn.Module):
    """
        Implements a model with two convolutional layers followed by pooling, and a final dense layer with 10 units.
    """

    def __init__(self, num_classes):
        super(FjordCifar10CNN, self).__init__()
        self.p = 1
        self.conv1 = nn.Conv2d(3, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.fc1 = nn.Linear(1280, num_classes)
   # def set_p(self , p):
    #    self.p = p
    def _masked(self, param, prev_param=None, dim=0):
        if dim == 0:
            _param = param[param.mask]
        else:
            _param = param[:, param.mask]
        if prev_param is not None:
            _param = _param[:, prev_param.mask]
        return _param

    def _get_mask(self, layer, dim=0, dropout_rate=1):
        N = layer.shape[dim]
        mask = np.zeros(N).astype(bool)
        mask[:int(N * dropout_rate)] = True
        return mask

    def compute_masks(self):
        self.conv1.weight.mask = self._get_mask(self.conv1.weight, dropout_rate=self.p)
        self.conv1.bias.mask = self._get_mask(self.conv1.bias, dropout_rate=self.p)

        self.conv2.weight.mask = self._get_mask(self.conv2.weight, dropout_rate=self.p)
        self.conv2.weight.prev_mask = self.conv1.weight.mask
        self.conv2.bias.mask = self._get_mask(self.conv2.bias, dropout_rate=self.p)

        self.fc1.weight.mask = self._get_mask(self.fc1.weight, dim=1, dropout_rate=self.p)
        #self.output.weight.mask = self._get_mask(self.output.weight, dim=1, dropout_rate=self.p)

    def forward(self, x, p=1):
        self.p = p
       # print(p)
        self.compute_masks()
        x = F.conv2d(
            x,
            self._masked(self.conv1.weight),
            self._masked(self.conv1.bias),
            1,  # stride
            2  # padding
        )
        x = self.pool(F.relu(x))
        # Second conv
        x = F.conv2d(
            x,
            self._masked(self.conv2.weight, self.conv1.weight),
            self._masked(self.conv2.bias),
            1,  # stride
            2  # padding
        )
        x = self.pool(F.relu(x))
        #print(x.size(0))
        x = x.view(x.size(0), -1)
        x = F.linear(x, self._masked(self.fc1.weight, dim=1), self.fc1.bias)
        return x

class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn = \
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


class FjordFemnistCNN2(nn.Module):
    """
        Implements a model with two convolutional layers followed by pooling, and a final dense layer with 10 units.
    """

    def __init__(self, num_classes):
        super(FjordFemnistCNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20 * 4 * 4, num_classes)


    def get_stop_idx(self, layer, type="conv", dropout_rate=None):
        if dropout_rate is None:
            stop_idx = -1
        else:
            if type == "conv":
                stop_idx = int(layer.weight.shape[0] * dropout_rate)
            elif type == "lin":
                stop_idx = int(layer.weight.shape[1] * dropout_rate)
        return stop_idx

    def forward(self, x, p=1):
        previous_stop_idx = x.shape[1]
        stop_idx = self.get_stop_idx(self.conv1, type="conv", dropout_rate=p)

        x = F.conv2d(
            x,
            self.conv1.weight[:stop_idx, :previous_stop_idx],
            self.conv1.bias[:stop_idx]
        )

        x = self.pool(F.relu(x))

        # Second conv
        previous_stop_idx = stop_idx
        stop_idx = self.get_stop_idx(self.conv2, type="conv", dropout_rate=p)
        x = F.conv2d(
            x,
            self.conv2.weight[:stop_idx, :previous_stop_idx],
            self.conv2.bias[:stop_idx]
        )
        x = self.pool(F.relu(x))

        x = self.flatten(x)

        stop_idx = self.get_stop_idx(self.fc1, type="lin", dropout_rate=p)
        x = F.linear(x, self.fc1.weight[:, :stop_idx], self.fc1.bias)

        return x


class MedMNISTCNN2(nn.Module):
    """
        Implements a model with two convolutional layers followed by pooling, and a final dense layer with 10 units.
    """


    def __init__(self, num_classes):
            super(MedMNISTCNN2, self).__init__()

            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU())

            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.layer3 = nn.Sequential(
                nn.Conv2d(16, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())

            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU())

            self.layer5 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))

            self.fc = nn.Sequential(
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes))


    # def set_p(self , p):
    #    self.p = p
    def _masked(self, param, prev_param=None, dim=0):
        if dim == 0:
            _param = param[param.mask]
        else:
            _param = param[:, param.mask]
        if prev_param is not None:
            _param = _param[:, prev_param.mask]
        return _param

    def _get_mask(self, layer, dim=0, dropout_rate=1):
        N = layer.shape[dim]
        mask = np.zeros(N).astype(bool)
        mask[:int(N * dropout_rate)] = True
        return mask

    def compute_masks(self):
        self.layer1[0].weight.mask = self._get_mask(self.layer1[0].weight, dropout_rate=self.p)
        self.layer2[0].weight.mask = self._get_mask(self.layer2[0].weight, dropout_rate=self.p)
        self.layer3[0].weight.mask = self._get_mask(self.layer3[0].weight, dropout_rate=self.p)
        self.layer4[0].weight.mask = self._get_mask(self.layer4[0].weight, dropout_rate=self.p)
        self.layer5[0].weight.mask = self._get_mask(self.layer5[0].weight, dropout_rate=self.p)


        self.fc[0].weight.mask = self._get_mask(self.fc[0].weight, dim=1, dropout_rate=self.p)
        # self.output.weight.mask = self._get_mask(self.output.weight, dim=1, dropout_rate=self.p)

    def forward(self, x, p=1):
        self.p = p
        # print(p)
       # self.compute_masks()


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


'''

class FjordFemnistCNN(nn.Module):
    """
        Implements a model with two convolutional layers followed by pooling, and a final dense layer with 10 units.
    """

    def __init__(self, num_classes):
        super(FjordFemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(20 * 4 * 4, num_classes)


    def get_stop_idx(self, layer, type="conv", dropout_rate=None):
        if dropout_rate is None:
            stop_idx = -1
        else:
            if type == "conv":
                stop_idx = int(layer.weight.shape[0] * dropout_rate)
            elif type == "lin":
                stop_idx = int(layer.weight.shape[1] * dropout_rate)
        return stop_idx

    def forward(self, x, p=1):
        previous_stop_idx = x.shape[1]
        stop_idx = self.get_stop_idx(self.conv1, type="conv", dropout_rate=p)

        x = F.conv2d(
            x,
            self.conv1.weight[:stop_idx],
            self.conv1.bias[:stop_idx]
        )
        x = self.pool(F.relu(x))

        # Second conv
        previous_stop_idx = stop_idx
        stop_idx = self.get_stop_idx(self.conv2, type="conv", dropout_rate=p)
        x = F.conv2d(
            x,
            self.conv2.weight[:stop_idx, :previous_stop_idx],
            self.conv2.bias[:stop_idx]
        )
        x = self.pool(F.relu(x))

        x = self.flatten(x)

        stop_idx = self.get_stop_idx(self.fc1, type="lin", dropout_rate=p)
        x = F.linear(x, self.fc1.weight[:, :stop_idx], self.fc1.bias)

        return x

'''
