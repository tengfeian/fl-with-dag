import torch
import random
import copy
from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer


# Exp 2. Centralized training with randomized order
def run_centralized_randomized_order(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device):
    # net = get_net(net_name)
    net = copy.deepcopy(initial_net)
    net.to(device)
    testloader = get_testloader(fds, dataset_name, batch_size)
    # define optimiser with hyperparameters supplied
    optim = get_optimizer(net)

    loss_list, accuracy_list = [],[]

    total_sequence = list(range(num_clients))*epochs
    random.seed(0)
    random.shuffle(total_sequence)
    # print(total_sequence)

    cnt = 0
    for partition_id in total_sequence:
        trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
        train(net, trainloader, optim, local_epochs, device)

        cnt += 1
        if cnt>0 and cnt%num_clients == 0:
            print(f"[Centralized training with randomized order] Training epoch {cnt//num_clients} ...")
            # evaluate model on the test set
            loss, accuracy = test(net, testloader, device)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print(f"[Centralized training with randomized order] accuracy list: {accuracy_list}")


    return loss_list, accuracy_list
