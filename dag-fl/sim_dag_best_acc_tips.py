from datafactory import get_trainloaders, get_testloader
from learning import train, test, get_optimizer
from dag_fl_alg import *
from aggregation import *
from dag_fl_select_tips import *
from utils import record_dag_info

from DAG import DAG
from client import Client
from DAGNode import DAGNode

# Exp 4. DAG-FL alg.
def run_dag_best_acc_tips(num_tips_selected, fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, device, all_round_to_clients, label):
    # Initialize DAG, lineage, GS tables
    dag = DAG()
    genesis_net = copy.deepcopy(initial_net)
    genesis = DAGNode(node_id=0, model=genesis_net, client_id=-1, creation_time=0, ref_nodes=[])
    dag.init(genesis)
    # Initialize clients
    clients = {}
    for client_id in range(num_clients):
        local_model = copy.deepcopy(genesis.model)
        clients[client_id] = Client(client_id, local_model,num_local_epoch, device)
        clients[client_id].trainloader = get_trainloaders(fds, client_id, dataset_name, batch_size)

    # For global model testing
    testloader = get_testloader(fds, dataset_name, batch_size)
    loss_list, accuracy_list = [],[]

    # In each round, new tips present in DAG, the clients who proposed these tips start to train new model
    # For each round,
    # 1. if e>0, update DAG by adding new tips, if e=0, pass
    # 2. each selected client chooses tips, beginning from e=0
    # 3. after tip selection, the clients train local model first
    # 4. selected clients set up DAGNode but keep it to itself without publishing it to DAG

    # todo the training in the last round will not be used to calculate the final global model, because we don't know when they will become tips
    global_node_nb = 0 # global node number for DAG
    for round in range(rounds):
        # todo delete
        print(f"[DAG Rand Tips alg.] Training round {round} ...")

        client_ids_current_round = all_round_to_clients[round]
        observed_num_clients = get_num_clients_observed(round, all_round_to_clients)

        tips= []
        if round > 0:
            # Set up DAG in the beginning of each round
            # Unbound tip nodes with their clients
            dag.add_tips(round, clients, client_ids_current_round)
            tips = dag.tips
            # Update DAG nodes in the beginning of each round
            for node in tips:
                # Compute lineage, GS for the DAGNode
                node.update_at_appearance(round, genesis, observed_num_clients)
            # Compute GS based on these tips
            dag.set_round_to_tip_gs(round)

        # Clients train local model
        for c_id in client_ids_current_round:
            c = clients[c_id]
            if round == 0:
                selected_tips = [dag.genesis]
            if round > 0:
                selected_tips = select_tips_by_accuracy(c, tips, num_tips_selected)
                # todo delete
                t_c_ids = []
                for t in selected_tips:
                    t_c_ids.append(t.client_id)
                print(f"round:{round}, c_id: {c_id},tips from clients:{t_c_ids}")

                c.update_local_model(selected_tips)
            c.local_train()

            # After training, create new DAGNode, but not attach to DAG
            global_node_nb += 1
            model = copy.deepcopy(c.local_model)
            # ref_nodes = copy.deepcopy(selected_tips)
            ref_nodes = selected_tips
            # Create node for this DAGNode
            node = DAGNode(global_node_nb, model, c_id, round, ref_nodes)
            c.dag_node = node


        # Evaluate model on the test set
        if round > 0:
            tip_models = [t.model.to(device) for t in tips]
            global_model = fedAvg(tip_models)

            loss, accuracy = test(global_model, testloader, device)
            loss_list.append(loss)
            accuracy_list.append(accuracy)
            print(f"[DAG Rand Tips alg.] accuracy list: {accuracy_list}")

    record_dag_info(dag, "dagRandTips_", label)
    return loss_list, accuracy_list

def get_num_clients_observed(round,all_round_to_clients):
    clients_appeared = set()
    # if round == 0:
    #     clients_appeared.update(all_round_to_clients[1])
    if round > 0:
        for i in range(1, round+1):
            clients_appeared.update(all_round_to_clients[i])

    return len(clients_appeared)