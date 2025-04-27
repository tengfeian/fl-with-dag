import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from datasets import load_dataset

def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        return load_dataset("ylecun/mnist")
    elif dataset_name == 'cifar10':
        return load_dataset("cifar10")
    else:
        raise NotImplementedError("Dataset {} not implemented".format(dataset_name))

def get_dataloaders(dataset_name, batch_size: int):
    dataset = get_dataset(dataset_name)

    if dataset_name == 'mnist':
        pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    elif dataset_name == 'cifar10':
        pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Prepare transformation functions
    def apply_transforms(batch):
        if dataset_name == 'mnist':
            key_name = "image"
        elif dataset_name == 'cifar10':
            key_name = "img"
        batch[key_name] = [pytorch_transforms(img) for img in batch[key_name]]
        return batch

    ds_train = dataset["train"].with_transform(apply_transforms)
    ds_test = dataset["test"].with_transform(apply_transforms)

    # Construct PyTorch dataloaders
    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(ds_test, batch_size=batch_size)
    return trainloader, testloader

