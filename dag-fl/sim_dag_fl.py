import copy
import random

from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *

# def models_equal(model1, model2):
#     return all(torch.equal(p1, p2) for p1, p2 in zip(model1.parameters(), model2.parameters()))


# Function to check if parameters are identical
def models_equal(model_a, model_b):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    # Ensure same keys
    if set(state_a.keys()) != set(state_b.keys()):
        return False

    # Compare each tensor
    for key in state_a:
        if not torch.equal(state_a[key], state_b[key]):
            print(f"Mismatch in parameter: {key}")  # Optional: for debugging
            return False

    return True

# Exp 4. DAG-FL alg.
def run_dag_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device, tips_all_rounds):
    # Initialize DAG, lineage, GS tables
    dag, lineage_cache, gs_cache = {},{},{}
    # Set initial model net into dag, lineage_cache, gs_cache
    genesis_model_id = 0
    update_dag(genesis_model_id, [], dag)
    update_lineage_cache(genesis_model_id, {}, lineage_cache)
    update_gs_cache(genesis_model_id, 0, gs_cache)

    tips_nets, incoming_tips_nets,  tips_optim = {},{},{}
    tip_node_model_id, incoming_tip_node_model_id = {},{}
    for partition_id in range(num_clients):
        tips_nets[partition_id] =  copy.deepcopy(initial_net)
        incoming_tips_nets[partition_id] =  copy.deepcopy(initial_net)
        # tips_optim[partition_id] = get_optimizer(initial_net) # bug has been fixed
        # tips_optim[partition_id] = get_optimizer(tips_nets[partition_id])
    testloader = get_testloader(fds, dataset_name, batch_size)

    clients_list = list(range(num_clients))
    loss_list, accuracy_list = [],[]
    model_id = genesis_model_id
    selected_tips, ref_model_ids = [], []
    for e in range(epochs):
        print(f"[DAG-FL alg.] Training epoch {e} ...")
        tips_current_round = tips_all_rounds[e]
        for partition_id in tips_current_round:
            # net = None
            if e == 0:
                # net = tips_nets[partition_id]
                net = incoming_tips_nets[partition_id]
            else: # Select tips, aggregate, and train
                print(
                    f"[run_dag_fl][select tips] partition_id: {partition_id}, tips_current_round: {tips_current_round},  ")
                selected_tips = select_tips(tips_current_round, partition_id, lineage_cache, dag, tip_node_model_id)
                model_list = [incoming_tips_nets[t] for t in selected_tips]
                net = fedAvg(model_list)
            net = net.to(device)
            optim = get_optimizer(net)
            # optim = tips_optim[partition_id]
            trainloader = get_trainloaders(fds, partition_id, dataset_name, batch_size)
            train(net, trainloader, optim, local_epochs, device)

            model_id += 1
            # Update the latest model id for the node
            incoming_tip_node_model_id[partition_id] = model_id
            # Update DAG
            if e ==0 : # All tips following genesis model point only to genesis model
                update_dag(model_id, [genesis_model_id], dag)
            else:
                ref_model_ids = [tip_node_model_id[t] for t in selected_tips]
                update_dag(model_id, ref_model_ids, dag)
            # Update lineage
            node_id = partition_id
            print(f"[run_dag_fl][complete one tip training] model_id: {model_id}, node_id: {node_id}, selected_tips:{selected_tips}, ref_model_ids:{ref_model_ids}  ")
            lineage = compute_lineage(model_id, node_id, dag, lineage_cache)
            update_lineage_cache(model_id, lineage, lineage_cache)
            # Update GS
            gs = compute_generalization_score(lineage,1)
            update_gs_cache(model_id, gs, gs_cache)

        # Update tips
        # todo delete
        for k in tips_nets.keys():
            equal = models_equal(tips_nets[k].to(device), incoming_tips_nets[k])
            # equal = models_equal(initial_net.to(device), incoming_tips_nets[k])
            print(f"###### e/1: {e}, k:{k} models are equals? "+ str(equal))

        tips_nets = copy.deepcopy(incoming_tips_nets)
        print(f"[run_dag_fl][Complete ALL tips trainings] tip_node_model_id: {tip_node_model_id}, incoming_tip_node_model_id: {incoming_tip_node_model_id}")
        tip_node_model_id = copy.deepcopy(incoming_tip_node_model_id)

        # todo delete
        for k in tips_nets.keys():
            equal = models_equal(tips_nets[k], incoming_tips_nets[k])
            # equal = models_equal(initial_net.to(device), incoming_tips_nets[k])
            print(f"###### e/2: {e}, k:{k} models are equals? "+ str(equal))

        # evaluate model on the test set
        model_list = [tips_nets[t] for t in clients_list]
        net = fedAvg(model_list)
        loss, accuracy = test(net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"[DAG-FL alg.] accuracy list: {accuracy_list}")

    return loss_list, accuracy_list

