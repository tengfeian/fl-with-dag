import copy
from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer


# Exp 1. Classic fully centralized sequential training
def run_classic_sequential(fds, initial_net, dataset_name, batch_size, rounds, local_epochs, num_clients, device, all_round_to_clients):
    net = copy.deepcopy(initial_net)
    net.to(device)
    testloader = get_testloader(fds, dataset_name, batch_size)
    # define optimiser with hyperparameters supplied
    optim = get_optimizer(net)

    loss_list, accuracy_list = [],[]
    for round in range(rounds):
        print(f"[Classic fully centralized sequential training] Training epoch {round} ...")
        for partition_id in range(num_clients):
            cur_round_clients = all_round_to_clients[round]
            if partition_id in cur_round_clients:
                trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
                train(net, trainloader, optim, local_epochs, device)
        # evaluate model on the test set
        loss, accuracy = test(net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"[Classic fully centralized sequential training] accuracy list: {accuracy_list}")

    return loss_list, accuracy_list

