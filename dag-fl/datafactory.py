from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

def split_dataset(dataset_name, partition_type, num_clients, dirichlet_alpha, seed):
    partioners = {"iid": IidPartitioner(num_partitions=num_clients),
                  "dirichlet": DirichletPartitioner(
                        num_partitions=num_clients,
                        partition_by="label",
                        alpha=dirichlet_alpha,
                        min_partition_size=0,
                        shuffle=True,
                        seed=seed
                        ),
                }
    partitioner = partioners[partition_type]

    if dataset_name == 'mnist':
        # return load_dataset("ylecun/mnist")
        return FederatedDataset(dataset="ylecun/mnist", partitioners={"train": partitioner})

    elif dataset_name == 'cifar10':
        # return load_dataset("cifar10")
        return FederatedDataset(dataset="uoft-cs/cifar10", partitioners={"train": partitioner})
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset_name))

def get_transforms(dataset_name):
    if dataset_name == 'mnist':
        pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    elif dataset_name == 'cifar10':
        pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        pytorch_transforms = None
    # Prepare transformation functions
    def apply_transforms(batch):
        if dataset_name == 'mnist':
            key_name = "image"
        elif dataset_name == 'cifar10':
            key_name = "img"
        else:
            key_name = None
        batch[key_name] = [pytorch_transforms(img) for img in batch[key_name]]
        return batch

    return apply_transforms

# partition_id equals to the node_id in clients
def get_trainloaders(fds, partition_id, dataset_name, batch_size):
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    # partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    apply_transforms = get_transforms(dataset_name)
    # Create train/val for each partition and wrap it into DataLoader
    # partition_train_test = partition_train_test.with_transform(apply_transforms)
    partition_train = partition.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train, batch_size=batch_size, shuffle=True)
    # valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader

def get_testloader(fds, dataset_name, batch_size):
    apply_transforms = get_transforms(dataset_name)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return testloader
