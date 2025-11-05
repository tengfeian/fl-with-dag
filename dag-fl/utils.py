import json
import torch
import copy

def write_logs(logfile, loss_list,accuracy_list, sim_name):
    # ensemble_log = args.log_file
    with open(logfile, mode="a") as file:
        file.write(sim_name+'\n')
        file.write(",".join(str(v) for v in accuracy_list))
        file.write('\n')
        file.write(",".join(str(v) for v in loss_list))
        file.write('\n')

def record_tips_info(file, rand_num_tips_all_rounds, all_round_to_clients, seed):
    with open(file, mode="a") as file:
        file.write("rand_num_tips_all_rounds with seed of "+str(seed) + ':\n[')
        json.dump(rand_num_tips_all_rounds, file)
        file.write('\n')
        file.write("all_round_to_clients with seed of " + str(seed) + ':\n')
        json.dump(all_round_to_clients, file)
        file.write('\n')

def record_dag_info(dag, sim,label):
    with open(sim+"DAG_node_id_to_ref_ids.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(dag.node_id_to_ref_ids, file)
        file.write('\n')
    with open(sim+"DAG_round_to_tip_gs.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(dag.round_to_tip_gs, file)
        file.write('\n')

    round_to_tip_ids = {}
    for r,tips in dag.round_to_tips.items():
        round_to_tip_ids[r] = [t.id for t in tips]
    with open(sim+"DAG_round_to_tips.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(round_to_tip_ids, file)
        file.write('\n')

    lineages = {}
    gs_all = {}
    cs_all = {}
    node_creation_appearance = {}
    for node_id, node in dag.node_id_to_node.items():
        lineages[node_id] = node.lineage
        gs_all[node_id] = node.gs
        cs_all[node_id] = node.cs
        node_creation_appearance[node_id] = [node.creation_time,node.appearance_time]
    with open(sim+"model_lineages.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(lineages, file)
        file.write('\n')
    with open(sim+"model_gs.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(gs_all, file)
        file.write('\n')
    with open(sim+"model_cs.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(cs_all, file)
        file.write('\n')
    with open(sim+"model_creation_appearance_time.json", mode="a") as file:
        file.write(label + ':\n')
        json.dump(node_creation_appearance, file)
        file.write('\n')

# Initialize clients after the previous simulation
def initilize_clients(initial_net, clients):
    for c_id in clients:
        local_model = copy.deepcopy(initial_net)
        clients[c_id].upload_local_model(local_model)

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

