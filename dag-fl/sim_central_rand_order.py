import random
import copy
from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from utils import initilize_clients


# Exp 2. Centralized training with randomized order
# def run_centralized_randomized_order(clients, testloader, initial_net, dataset_name, batch_size, rounds, local_epochs, num_clients, device, all_round_to_clients):
def run_centralized_randomized_order(clients, testloader, initial_net, num_local_epoch, num_clients, device, all_round_to_clients):
    net = copy.deepcopy(initial_net)
    net.to(device)
    # testloader = get_testloader(fds, dataset_name, batch_size)
    # Define optimiser with hyperparameters supplied
    optim = get_optimizer(net)

    # Initialize clients
    # todo unnecessary, in fact the clients' local model is not used here
    # initilize_clients(initial_net, clients)

    loss_list, accuracy_list = [],[]

    # total_sequence = list(range(num_clients))*rounds
    total_sequence = [n for each_round in all_round_to_clients for n in each_round]
    random.seed(0)
    random.shuffle(total_sequence)

    cnt, r = 0, 0
    for partition_id in total_sequence:
        # trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
        trainloader = clients[partition_id].trainloader
        train(net, trainloader, optim, num_local_epoch, device)

        cnt += 1
        # if cnt>0 and cnt%num_clients == 0:
        if cnt == len(all_round_to_clients[r]):
            print(f"[Centralized training with randomized order] Training round {cnt//num_clients} ...")
            # Evaluate model on the test set
            loss, accuracy = test(net, testloader, device)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print(f"[Centralized training with randomized order] accuracy list: {accuracy_list}")
            cnt = 0
            r +=1


    return loss_list, accuracy_list
