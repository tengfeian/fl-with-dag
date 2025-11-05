import copy
import random

from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *
from utils import initilize_clients

# Exp 3. Traditional syn/async FL
# def run_traditional_fl(clients, testloader, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device, nodes_for_aggregation_all_rounds=None):
def run_traditional_fl(clients, testloader, initial_net, rounds, num_clients, device, nodes_for_aggregation_all_rounds=None):
    # For sync fl, nodes_for_aggregation_all_rounds = None, for async fl, nodes_for_aggregation_all_rounds is a list of list
    if nodes_for_aggregation_all_rounds is None:
        nodes_for_aggregation_all_rounds = []
    # client_nets = {}
    # for partition_id in range(num_clients):
    #     client_nets[partition_id] =  copy.deepcopy(initial_net).to(device)
    # testloader = get_testloader(fds, dataset_name, batch_size)

    # clients_list = list(range(num_clients))

    # Initialize clients
    initilize_clients(initial_net, clients)

    loss_list, accuracy_list = [],[]
    for round in range(rounds):
        if not nodes_for_aggregation_all_rounds:
            print(f"[Traditional synchronous FL] Training epoch {round} ...")
            nodes_for_aggregation = list(range(num_clients))
        else:
            print(f"[Traditional Asynchronous FL] Training epoch {round} ...")
            nodes_for_aggregation = nodes_for_aggregation_all_rounds[round]

        # for partition_id in nodes_for_aggregation:
        #     net = client_nets[partition_id]
        #     optim = get_optimizer(net)
        #     trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
        #     train(net, trainloader, optim, local_epochs, device)

        for c_id in nodes_for_aggregation:
            c = clients[c_id]
            c.local_train()

        # Aggregate model and evaluate model on the test set
        model_list = [clients[c_id].local_model for c_id in nodes_for_aggregation]
        # model_list = [client_nets[c] for c in nodes_for_aggregation]
        aggregated_net = fedAvg(model_list).to(device)
        loss, accuracy = test(aggregated_net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)

        # Disseminate aggregated model to all clients
        for c_id in range(num_clients):
            # client_nets[partition_id] = copy.deepcopy(aggregated_net).to(device)
            clients[c_id].upload_local_model(aggregated_net)
        if not nodes_for_aggregation_all_rounds:
            print(f"[Traditional synchronous FL] accuracy list: {accuracy_list}")
        else:
            print(f"[Traditional Asynchronous FL] accuracy list: {accuracy_list}")

    return loss_list, accuracy_list

