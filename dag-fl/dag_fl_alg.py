import copy
import numpy as np
import random

"""
Data structure:
    dag = {model_id: [reference_model_id,...]}
    lineage = {node_id: values}
    lineage_cache = {model_id: lineage}
"""

def get_lineages(model_ids, lineage_cache):
    lineages = []
    print(f"[get_lineages] model_ids {model_ids}")
    for model_id in model_ids:
        lineages.append(lineage_cache[model_id])
    return lineages

def update_dag(model_id, ref_model_ids, dag):
    if model_id in dag:
        raise ValueError('model_id already present in dag')
    dag[model_id] = ref_model_ids

def update_lineage_cache(model_id, lineage, lineage_cache):
    if model_id in lineage_cache:
        raise ValueError('model_id already present in lineage_cache')
    lineage_cache[model_id] = lineage

def update_gs_cache(model_id, gs, gs_cache):
    if model_id in gs_cache:
        raise ValueError('model_id already present in gs_cache')
    gs_cache[model_id] = gs

def fusion(lineages):
    """
    Merges multiple lineages while ensuring correct normalization.
    Each node's contribution is divided by the number of referenced lineages.
    lineage: {k:v}
    """
    merged_lineage = {}
    num_references = len(lineages)

    for lineage in lineages:
        for node, value in lineage.items():
            if node in merged_lineage:
                merged_lineage[node] += value
            else:
                merged_lineage[node] = value

    if num_references > 0:
        for node in merged_lineage:
            merged_lineage[node] /= num_references

    return merged_lineage

def compute_lineage(model_id, node_id, dag, lineage_cache):
    """
    Computes the lineage of a model while ensuring correct normalization.
    Being called after the model has added to dag as a tip.
    """
    if model_id == 0:
        return {}

    # At epoch 0, dag is empty, so return [0] for the models in epoch 0
    ref_model_ids = dag[model_id] if model_id in dag else [0]

    # If model references is the Genesis
    if ref_model_ids == [0]:
        return {node_id: 1}
    print(f"[compute_lineage] model_id:{model_id}, node_id:{node_id}, ref_model_ids: {ref_model_ids}")

    reference_lineages = get_lineages(ref_model_ids,lineage_cache)
    merged_lineage = fusion(reference_lineages)
    # Avoid the case for virtual global model GS calculation in tips selection alg.
    if node_id >= 0:
        merged_lineage[node_id] = merged_lineage.get(node_id, 0) + 1
    return merged_lineage

def compute_generalization_score(lineage, epoch, tipnodes_all_rounds, a=1):
    """
    Computes the Generalization Score (M) for each model.
    The number of nodes has presented in DAG up to this epoch is based on the observation of previous epoch
    The 0th epoch is special, we assume its observation is based on the 0th epoch's tips presence.
    """
    nodes_presented = set()
    if epoch == 0:
        nodes_presented.update(tipnodes_all_rounds[1])
    else:
        for i in range(1, epoch+1):
            nodes_presented.update(tipnodes_all_rounds[i])

    observed_num_nodes = len(nodes_presented)
    # todo delete
    print(f"nodes_presented: {nodes_presented}")
    non_zero_nodes = sum(1 for value in lineage.values() if value > 0)
    if len(lineage) > 1:
        std_lineage = np.std(list(lineage.values()))
    else:
        std_lineage = 0
    v = non_zero_nodes / (observed_num_nodes * (1 + a * std_lineage))
    return v

# Tips refer to the model id
# todo: some very old tips should be excluded from selection
def select_tips(tip_nets, node_id, lineage_cache, dag, epoch, tipnodes_all_rounds):
    selected_tips = [] # Store model_id of tips
    max_virtual_final_GS = 0
    virtual_tips = list(tip_nets.keys())
    vitual_dag = copy.deepcopy(dag)
    virtual_lineage_cache = copy.deepcopy(lineage_cache)

    random_tip = random.choice(virtual_tips)
    selected_tips.append(random_tip)
    virtual_tips.remove(random_tip)

    virtual_finalmodel_id = 10 ** 12
    virtual_model_id = 9**12

    for tip in virtual_tips:
        selected_tips.append(tip)
        virtual_tips.remove(tip)
        ref_model_ids = selected_tips
        # Add the virtual tip to temp DAG
        update_dag(virtual_model_id, ref_model_ids, vitual_dag)

        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_lineage = compute_lineage(virtual_model_id, node_id, vitual_dag, virtual_lineage_cache)
        update_lineage_cache(virtual_model_id, virtual_lineage, virtual_lineage_cache)

        # Add virtual final model to the virtual DAG
        original_tips = list(tip_nets.keys())
        unused_tips = [t for t in original_tips if t not in selected_tips]
        ref_model_ids_finalmodel = unused_tips + [virtual_model_id]
        update_dag(virtual_finalmodel_id, ref_model_ids_finalmodel, vitual_dag)
        # Calculate lineage for virtual global model
        virtual_final_mode_node_id = -1
        virtual_final_lineage = compute_lineage(virtual_finalmodel_id, virtual_final_mode_node_id, vitual_dag, virtual_lineage_cache)
        update_lineage_cache(virtual_finalmodel_id, virtual_final_lineage, virtual_lineage_cache)

        # Calculate gs for virtual global model
        virtual_final_GS = compute_generalization_score(virtual_final_lineage, epoch, tipnodes_all_rounds)
        if virtual_final_GS > max_virtual_final_GS:
            max_virtual_final_GS = virtual_final_GS
        else:
            selected_tips.remove(tip)
        del vitual_dag[virtual_model_id]
        del vitual_dag[virtual_finalmodel_id]
        del virtual_lineage_cache[virtual_model_id]
        del virtual_lineage_cache[virtual_finalmodel_id]

    return selected_tips