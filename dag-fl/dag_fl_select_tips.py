import copy
import random
from DAGNode import DAGNode

# Tips refer to the DAGNode
# todo: to consider to exclude some very old tips from selection
# def select_tips(tip_nets, node_id, lineage_cache, dag, epoch, tipnodes_all_rounds):
def select_tips(tips, client_id, genesis, round, observed_num_clients):
    # # todo delete
    # tip_ids = []
    # for t in tips:
    #     tip_ids.append(t.id)
    # print(f"[select_tips]: client_id = {client_id}, round = {round}, observed_num_clients = {observed_num_clients}, tip_ids: {tip_ids}")

    selected_tips = []
    max_virtual_final_gs = 0
    # original_tips = copy.copy(tips)
    virtual_tips = copy.copy(tips)

    random.shuffle(virtual_tips)
    random_tip = virtual_tips.pop(0)
    # random_tip = random.choice(virtual_tips)
    selected_tips.append(random_tip)
    # virtual_tips.remove(random_tip)

    virtual_final_node_id = -2
    virtual_node_id = 9**12

    for tip in virtual_tips:
        selected_tips.append(tip)
        # virtual_tips.remove(tip)
        # ref_model_ids = selected_tips
        virtual_ref_nodes = selected_tips

        virtual_node = DAGNode(virtual_node_id, None, client_id, round, virtual_ref_nodes)
        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_node.set_lineage(genesis)

        unused_virtual_tips = [t for t in virtual_tips if t not in selected_tips]
        # unused_virtual_tips = [t for t in original_tips if t not in selected_tips]
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

def select_tips_by_accuracy(client, tips, num_tips_selected):
    tip_acc = []
    for tip in tips:
        _, acc = client.local_test(tip.model)
        tip_acc.append((tip,acc))
    tip_acc.sort(key=lambda x: x[1], reverse=True)
    tips_selected = [ta[0] for ta in tip_acc[:num_tips_selected]]
    return tips_selected

def select_tips_sMSA_v2(tips, client_id, genesis, round, observed_num_clients):
    # # todo delete
    # tip_ids = []
    # for t in tips:
    #     tip_ids.append(t.id)
    # print(f"[select_tips]: client_id = {client_id}, round = {round}, observed_num_clients = {observed_num_clients}, tip_ids: {tip_ids}")

    selected_tips = []
    # max_virtual_final_gs = 0
    max_virtual_node_gs = 0
    # original_tips = copy.copy(tips)
    virtual_tips = copy.copy(tips)

    random.shuffle(virtual_tips)
    random_tip = virtual_tips.pop(0)
    # random_tip = random.choice(virtual_tips)
    selected_tips.append(random_tip)
    # virtual_tips.remove(random_tip)

    virtual_final_node_id = -2
    virtual_node_id = 9**12

    for tip in virtual_tips:
        selected_tips.append(tip)
        # virtual_tips.remove(tip)
        # ref_model_ids = selected_tips
        virtual_ref_nodes = selected_tips

        virtual_node = DAGNode(virtual_node_id, None, client_id, round, virtual_ref_nodes)
        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_node.set_lineage(genesis)
        virtual_node.set_gs(observed_num_clients)
        if virtual_node.gs > max_virtual_node_gs:
            max_virtual_node_gs = virtual_node.gs
        else:
            selected_tips.remove(tip)

    # unused_virtual_tips = [t for t in original_tips if t not in selected_tips]
    unused_virtual_tips = [t for t in virtual_tips if t not in selected_tips]
    virtual_final_ref_nodes = unused_virtual_tips + [virtual_node]
    virtual_final_node = DAGNode(virtual_final_node_id, None, -1, round, virtual_final_ref_nodes)
    # Calculate lineage for this virtual tip and add it to temp lineage cache
    virtual_final_node.set_lineage(genesis)
    # todo disputing, if we should set all number of clients
    virtual_final_node.set_gs(observed_num_clients)

    # if virtual_final_node.gs > max_virtual_final_gs:
    #     max_virtual_final_gs = virtual_final_node.gs
    # else:
    #     selected_tips.remove(tip)

    return selected_tips, virtual_final_node.gs