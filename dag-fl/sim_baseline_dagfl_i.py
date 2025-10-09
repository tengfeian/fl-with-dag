from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *
from utils import record_dag_info

def select_rand_tips(tip_nets):
    tip_model_ids = list(tip_nets.keys())
    # todo hard-coded for two elements to be selected
    return random.sample(tip_model_ids, 2)

# Exp 4. Baseline DAG-FL I
def run_baseline_dagfl_i(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device, tipnodes_all_rounds, seed):
    dag = {}
    genesis_model_id = 0
    node_nets, incoming_node_nets = {},{}
    model_birth, model_presence, tip_nets = {},{},{}
    node_model_id, incoming_node_model_id = {},{}
    model_references = {}
    invalide_tips = set()
    # new_model_nets = {}

    for node_id in range(num_clients):
        incoming_node_nets[node_id] =  copy.deepcopy(initial_net)
        # tips_optim[node_id] = get_optimizer(initial_net) # bug has been fixed in below
        # tips_optim[node_id] = get_optimizer(node_nets[node_id])
    testloader = get_testloader(fds, dataset_name, batch_size)

    clients_list = list(range(num_clients))
    loss_list, accuracy_list = [],[]
    model_id = genesis_model_id
    selected_tips = []
    for e in range(epochs):
        tipnodes_current_round = tipnodes_all_rounds[e]
        if e > 0:
            for node_id in tipnodes_current_round:
                m_id = node_model_id[node_id]
                if e == 1:  # All tips following the genesis model point only to genesis model
                    ref_model_ids = [genesis_model_id]
                elif e>1:
                    ref_model_ids = model_references[m_id]
                # Update DAG, this time represents when the model of previous epoch show up in DAG
                update_dag(m_id, ref_model_ids, dag)
                # Set up model_id and nets for tips in this epoch
                net_ = node_nets[node_id]
                tip_nets[m_id] = net_
                # Set up the presence time of models
                model_presence[m_id] = e

        for node_id in tipnodes_current_round:
            if e == 0:
                net = incoming_node_nets[node_id]
            else: # Select tips and aggregate them, and train it
                selected_tips = select_rand_tips(tip_nets)
                model_list = [tip_nets[t] for t in selected_tips]
                net = fedAvg(model_list)
            net = net.to(device)
            optim = get_optimizer(net)
            trainloader = get_trainloaders(fds, node_id, dataset_name, batch_size)
            train(net, trainloader, optim, local_epochs, device)

            model_id += 1
            # Update the latest model id and models for the nodes and tips
            incoming_node_model_id[node_id] = model_id
            incoming_node_nets[node_id] = net
            invalide_tips.update(selected_tips)
            # new_model_nets[model_id] = net
            model_birth[model_id] = e
            model_references[model_id] = selected_tips

        # Update current node-net-model_id information after one global epoch
        node_nets = copy.deepcopy(incoming_node_nets)
        print(f"[run_baseline_dagfl_i][Complete ALL tips trainings] node_model_id: {node_model_id}, incoming_node_model_id: {incoming_node_model_id}")
        node_model_id = copy.deepcopy(incoming_node_model_id)
        for t in invalide_tips:
            # del model_birth[t]
            del tip_nets[t]
        invalide_tips.clear()

        # Evaluate model on the test set
        model_list = [node_nets[t] for t in clients_list]
        global_net = fedAvg(model_list)

        loss, accuracy = test(global_net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"[Baseline DAG-FL I] accuracy list: {accuracy_list}")

    record_dag_info(dag, {}, {}, seed, model_presence, model_birth, "Baseline DAG-FL I")


    return loss_list, accuracy_list