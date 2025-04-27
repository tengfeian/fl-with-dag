import torch
from nets import get_net
from datafactory import get_dataloaders
from learning import train, test, get_optimizer

def run_centralised(net_name, dataset_name, batch_size, epochs: int, lr: float, momentum:float, device):
    model = get_net(net_name)
    model.to(device)

    trainloader, testloader = get_dataloaders(dataset_name, batch_size)

    # define optimiser with hyperparameters supplied
    optim = get_optimizer(model, lr, momentum)

    # train for the specified number of epochs
    loss_list, accuracy_list = [],[]
    for e in range(epochs):
        print(f"[Centralized learning] Training epoch {e} ...")
        train(model, trainloader, optim, device)
        # evaluate model on the test set
        loss, accuracy = test(model, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"accuracy list: {accuracy_list}")

    return loss_list, accuracy_list
