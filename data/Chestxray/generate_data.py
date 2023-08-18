"""
Download EMNIST dataset, and splits it among clients
"""
import os
import argparse
import pickle
import os
import glob
from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset
from PIL import Image
from sklearn.model_selection import train_test_split

from utils import split_dataset_by_labels, pathological_non_iid_split
from torch.utils.data import Dataset, DataLoader, random_split

# TODO: remove this after new release of torchvision
EMNIST.url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

N_CLASSES = 2
RAW_DATA_PATH = "raw_data/"
PATH = "all_data/"


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; '
             'default is 0.2',
        type=float,
        default=0.2)
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 0.2;',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction in validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=1234
    )

    return parser.parse_args()


def main():
    args = parse_args()

    transform = Compose(
        [ToTensor(),
         Normalize((0.1307,), (0.3081,))
         ]
    )

    dataset = ConcatDataset([
         ChestXray('/Users/ali/PycharmProjects/fjord-green/raw_data/Chestxray/chestXray/train', transform=transform),
     ChestXray('/Users/ali/PycharmProjects/fjord-green/raw_data/Chestxray/chestXray/test', transform=transform)
    ])
    print(len(dataset))
    if args.pathological_split:
        clients_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed,
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(PATH, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            client_path = os.path.join(PATH, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)
            print(client_id)
            train_indices, test_indices =\
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1.-args.val_frac,
                        random_state=args.seed
                    )

                save_data(val_indices, os.path.join(client_path, "val.pkl"))

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))



class ChestXray(Dataset):
    def __init__(self, root, transform, ext='jpeg'):
        """ Instantiate an object of the FlowerDataset
        Args:
          root (str): The root directory includes images, where each class of images is
            inside a separate folder.
          transformation (callable): A image augmentation function to be applied to an image.
            default is None, indicating the there will be no transformation applied
            an image.
        """
        self.paths = []
        self.labels = []

        # In this assignment label encoder was given.
        # However, if we wanted to make it ourself,
        self.label_encoder = {cat: index for index, cat in \
                              enumerate(cat for cat in os.listdir(root) if \
                                        os.path.isdir(os.path.join(root, cat)) and \
                                        not cat.startswith('.'))}
        # end of label encoder

        self.label_encoder = {"NORMAL": 0, "PNEUMONIA": 1}
        # TODO: Read the path to all images and indicat their classes.
        # Hint: You can use glob.glob(f'{directory_path}/**/*.jpg', recursive=True)
        #       to read all images with .jpg file extension inside a directory_path
        #       Also, you can use te self.label_encoder to assign a numerical label
        #       to each flower class

        self.transform = transform
        self.label_decoder = {index: cat for cat, index in self.label_encoder.items()}

        for cat in self.label_encoder.keys():
            category_path = glob.glob(os.path.join(root, cat, f'*.{ext}'))
            self.paths += category_path
            self.labels += [self.label_encoder[cat]] * len(category_path)

        # End of TODO
        assert len(self.paths) == len(self.labels), "Number of image paths and labels should be equal"

    def __len__(self):
        """ Return the number of samples within the dataset """
        # TODO: replace return 0 with the proper code for implementing the requested functionality
        return len(self.paths)

    def __getitem__(self, i):
        """ This method return a tuple made of the i-th element of the dataset, i.e. (image, label).

        If the transofrm is not None, you should apply the transform function to the image."""
        # TODO: write code for implementing the requested functionality
        image = Image.open(self.paths[i])
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[i]
        return image, label
if __name__ == "__main__":
    main()
