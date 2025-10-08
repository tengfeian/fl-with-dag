import copy

import numpy as np
import random

# dag = {model_id: [reference_model_id,...]}
# lineage_cache = {model_id: lineage}
# lineage = {node_id: values}

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
    # if model_id in lineage_cache:
    #     return lineage_cache[model_id]

    if model_id == 0:
        return {}

    ref_model_ids = dag[model_id] if model_id in dag else []
    # If model references is the Genesis
    # if not ref_model_ids:
    if ref_model_ids == [0]:
        return {node_id: 1}
    print(f"[compute_lineage] model_id:{model_id}, node_id:{node_id}, ref_model_ids: {ref_model_ids}")

    reference_lineages = get_lineages(ref_model_ids,lineage_cache)
    merged_lineage = fusion(reference_lineages)
    if node_id >= 0: # Avoid the case for virtual global model GS calculation in tips selection alg.
        merged_lineage[node_id] = merged_lineage.get(node_id, 0) + 1
    return merged_lineage

def compute_generalization_score(lineage, a=1):
    """
    Computes the Generalization Score (M) for each model.
    """
    # TODO Observed_num_nodes and non_zero_nodes almost identical
    observed_num_nodes = len(lineage)
    non_zero_nodes = sum(1 for value in lineage.values() if value > 0)
    if len(lineage) > 1:
        std_lineage = np.std(list(lineage.values()))
    else:
        std_lineage = 0
    v = non_zero_nodes / (observed_num_nodes * (1 + a * std_lineage))
    return v

# tips = [model_id,...]
def select_tips(tips, node_id,lineage_cache, dag, tip_node_model_id):
    selected_tips = []
    max_virtual_final_GS = 0
    virtual_tips = tips.copy()
    vitual_dag = copy.deepcopy(dag)
    virtual_lineage_cache = copy.deepcopy(lineage_cache)

    random_tip = random.choice(tips)
    selected_tips.append(random_tip)
    virtual_tips.remove(random_tip)

    virtual_final_model_id = 10 ** 10
    virtual_model_id = 9**10

    for tip in virtual_tips:
        selected_tips.append(tip)
        virtual_tips.remove(tip)
        ref_model_ids = [tip_node_model_id[t] for t in selected_tips]
        # Add the virtual tip to temp DAG
        update_dag(virtual_model_id, ref_model_ids, vitual_dag)

        # Calculate lineage for this virtual tip and add it to temp lineage cache
        virtual_lineage = compute_lineage(virtual_model_id, node_id, vitual_dag, virtual_lineage_cache)
        update_lineage_cache(virtual_model_id, virtual_lineage, virtual_lineage_cache)

        # Add virtual final model to the virtual DAG
        unused_tips = [t for t in tips if t not in selected_tips]
        ref_model_ids = [tip_node_model_id[t] for t in unused_tips] + [virtual_model_id]
        update_dag(virtual_final_model_id, ref_model_ids, vitual_dag)
        # Calculate lineage for virtual global model
        virtual_final_mode_node_id = -1
        virtual_final_lineage = compute_lineage(virtual_final_model_id, virtual_final_mode_node_id, vitual_dag, virtual_lineage_cache)
        update_lineage_cache(virtual_final_model_id, virtual_final_lineage, virtual_lineage_cache)

        # Calculate gs for virtual global model
        virtual_final_GS = compute_generalization_score(virtual_final_lineage)
        if virtual_final_GS > max_virtual_final_GS:
            max_virtual_final_GS = virtual_final_GS
        else:
            selected_tips.remove(tip)
        del vitual_dag[virtual_model_id]
        del vitual_dag[virtual_final_model_id]
        del virtual_lineage_cache[virtual_model_id]
        del virtual_lineage_cache[virtual_final_model_id]

    return selected_tips