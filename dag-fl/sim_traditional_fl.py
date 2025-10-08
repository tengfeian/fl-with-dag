import copy
import random

from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *

# Exp 3. Traditional syn/async FL
def run_traditional_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device, nodes_for_aggregation_all_rounds=None):
    if nodes_for_aggregation_all_rounds is None:
        nodes_for_aggregation_all_rounds = []
    client_nets = {}
    for partition_id in range(num_clients):
        client_nets[partition_id] =  copy.deepcopy(initial_net).to(device)
    testloader = get_testloader(fds, dataset_name, batch_size)

    clients_list = list(range(num_clients))
    loss_list, accuracy_list = [],[]
    for e in range(epochs):
        if not nodes_for_aggregation_all_rounds:
            print(f"[Traditional synchronous FL] Training epoch {e} ...")
            nodes_for_aggregation = list(range(num_clients))
        else:
            print(f"[Traditional Asynchronous FL] Training epoch {e} ...")
            nodes_for_aggregation = nodes_for_aggregation_all_rounds[e]

        for partition_id in nodes_for_aggregation:
            net = client_nets[partition_id]
            optim = get_optimizer(net)
            trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
            train(net, trainloader, optim, local_epochs, device)

        # Aggregate model and evaluate model on the test set
        model_list = [client_nets[c] for c in nodes_for_aggregation]
        aggregated_net = fedAvg(model_list).to(device)
        loss, accuracy = test(aggregated_net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)

        # Disseminate aggregated model to all clients
        for partition_id in range(num_clients):
            client_nets[partition_id] = copy.deepcopy(aggregated_net).to(device)

        if not nodes_for_aggregation_all_rounds:
            print(f"[Traditional synchronous FL] accuracy list: {accuracy_list}")
        else:
            print(f"[Traditional Asynchronous FL] accuracy list: {accuracy_list}")

    return loss_list, accuracy_list

