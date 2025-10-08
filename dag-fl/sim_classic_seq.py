import copy
from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer


# Exp 1. Classic fully centralized sequential training
def run_classic_sequential(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device):
    # net = get_net(net_name)
    net = copy.deepcopy(initial_net)
    net.to(device)
    testloader = get_testloader(fds, dataset_name, batch_size)
    # define optimiser with hyperparameters supplied
    optim = get_optimizer(net)

    loss_list, accuracy_list = [],[]
    for e in range(epochs):
        print(f"[Classic fully centralized sequential training] Training epoch {e} ...")
        for partition_id in range(num_clients):
            trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
            train(net, trainloader, optim, local_epochs, device)
        # evaluate model on the test set
        loss, accuracy = test(net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"[Classic fully centralized sequential training] accuracy list: {accuracy_list}")

    return loss_list, accuracy_list

