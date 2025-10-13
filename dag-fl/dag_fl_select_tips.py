import copy
import random
from DAGNode import DAGNode

# Tips refer to the model id
# todo: some very old tips should be excluded from selection
# def select_tips(tip_nets, node_id, lineage_cache, dag, epoch, tipnodes_all_rounds):
def select_tips(tips, client_id, genesis, round, observed_num_clients):
    # # todo delete
    # tip_ids = []
    # for t in tips:
    #     tip_ids.append(t.id)
    # print(f"[select_tips]: client_id = {client_id}, round = {round}, observed_num_clients = {observed_num_clients}, tip_ids: {tip_ids}")

    selected_tips = []
    max_virtual_final_gs = 0
    original_tips = copy.copy(tips)
    virtual_tips = copy.copy(tips)

    random_tip = random.choice(virtual_tips)
    selected_tips.append(random_tip)
    virtual_tips.remove(random_tip)

    virtual_final_node_id = -2
    virtual_node_id = 9**12

    for tip in virtual_tips:
        selected_tips.append(tip)
        virtual_tips.remove(tip)
        # ref_model_ids = selected_tips
        virtual_ref_nodes = selected_tips

        virtual_node = DAGNode(virtual_node_id, None, client_id, round, virtual_ref_nodes)
        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_node.set_lineage(genesis)

        unused_virtual_tips = [t for t in original_tips if t not in selected_tips]
        virtual_final_ref_nodes = unused_virtual_tips + [virtual_node]
        virtual_final_node = DAGNode(virtual_final_node_id, None, -1, round, virtual_final_ref_nodes)
        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_final_node.set_lineage(genesis)
        # todo disputing, if we should set all number of clients
        virtual_final_node.set_gs(observed_num_clients)

        if virtual_final_node.gs > max_virtual_final_gs:
            max_virtual_final_gs = virtual_final_node.gs
        else:
            selected_tips.remove(tip)

    return selected_tips, max_virtual_final_gs