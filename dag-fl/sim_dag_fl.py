from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *
from utils import record_dag_info

# Exp 4. DAG-FL alg.
def run_dag_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, device, tipnodes_all_rounds, seed):
    # Initialize DAG, lineage, GS tables
    dag, lineage_cache, gs_cache = {},{},{}
    # Set initial model net into dag, lineage_cache, gs_cache
    genesis_model_id = 0
    update_dag(genesis_model_id, [], dag)
    update_lineage_cache(genesis_model_id, {}, lineage_cache)
    update_gs_cache(genesis_model_id, 0, gs_cache)

    """
    Data structure:
        node_nets: format {node_id: net}, records the latest model each node holds before new tips coming.
        incoming_node_nets: format {node_id: net}, records the latest model each node holds during the coming of new tips when node net may be replaced by new one.
        node_model_id: format {node_id: model_id}, records the latest model id each node holds before new tips coming.
        incoming_node_model_id: format {node_id: net},  records the latest model id each node holds during the coming of new tips when node net may be replaced by new one.
        model_birth: format {model_id: int}, records the birth epoch of model in each epoch, for analysis
        model_presence: format {model_id: int}, records the presence of model_id in each epoch, for analysis
        tip_nets: format {model_id: net}, records the tips' model_id and net in current epoch.
        model_references: format {model_id: net}, records all reference models for all models at their birth time, note that this new model may be still in computing and does not present in the DAG until the epoch its node is selected by "tipnodes_current_round".
    Update DAG logics:
        Update DAG and GS cache occurs at the beginning of each epoch, i.e. the time presence of model in DAG.
        Update lineage cache occurs at the birth of new model.
    """
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
        # todo delete
        print(f"[DAG-FL alg.] Training epoch {e} ...")
        tipnodes_current_round = tipnodes_all_rounds[e]

        # todo delete
        print(f"tipnodes_current_round:{tipnodes_current_round}")

        if e > 0:
            for node_id in tipnodes_current_round:
                m_id = node_model_id[node_id]
                if e == 1:  # All tips following the genesis model point only to genesis model
                    ref_model_ids = [genesis_model_id]
                elif e>1:
                    ref_model_ids = model_references[m_id]
                # Update DAG, this time represents when the model of previous epoch show up in DAG
                update_dag(m_id, ref_model_ids, dag)
                # Update lineage
                print(f"lineage_cache:{lineage_cache}")
                print(f"m_id:{m_id}")
                lineage = compute_lineage(m_id, node_id, dag, lineage_cache)
                update_lineage_cache(m_id, lineage, lineage_cache)
                # Update GS
                lineage = lineage_cache[m_id]
                gs = compute_generalization_score(lineage, e, tipnodes_all_rounds)
                update_gs_cache(m_id, gs, gs_cache)
                # Set up model_id and nets for tips in this epoch
                net_ = node_nets[node_id]
                tip_nets[m_id] = net_
                # Set up the presence time of models
                model_presence[m_id] = e

        for node_id in tipnodes_current_round:
            if e == 0:
                net = incoming_node_nets[node_id]
            else: # Select tips and aggregate them, and train it
                # todo delete
                print(f"[run_dag_fl][select tips] node_id: {node_id}, tipnodes_current_round: {tipnodes_current_round},  ")
                selected_tips = select_tips(tip_nets, node_id, lineage_cache, dag, e, tipnodes_all_rounds)
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

            # todo delete
            print(f"[run_dag_fl][complete one tip training] model_id: {model_id}, node_id: {node_id}, selected_tips:{selected_tips}, ref_model_ids:{model_references[model_id]}  ")

        # Update current node-net-model_id information after one global epoch
        node_nets = copy.deepcopy(incoming_node_nets)
        print(f"[run_dag_fl][Complete ALL tips trainings] node_model_id: {node_model_id}, incoming_node_model_id: {incoming_node_model_id}")
        node_model_id = copy.deepcopy(incoming_node_model_id)
        # for t in new_model_nets.keys():
        #     model_birth[t] = e
        #     potential_tip_nets[t] = new_model_nets[t]
        for t in invalide_tips:
            # del model_birth[t]
            del tip_nets[t]
        invalide_tips.clear()
        # new_model_nets.clear()

        # Evaluate model on the test set
        model_list = [node_nets[t] for t in clients_list]
        global_net = fedAvg(model_list)

        loss, accuracy = test(global_net, testloader, device)
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        print(f"[DAG-FL alg.] accuracy list: {accuracy_list}")

    record_dag_info(dag, lineage_cache, gs_cache, seed, model_presence, model_birth, "DAG-FL with GS")

    return loss_list, accuracy_list

