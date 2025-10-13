import numpy as np

def fusion(lineages):
    """
    Merges multiple lineages while ensuring correct normalization.
    Each node's contribution is divided by the number of referenced lineages.
    lineage: {k:v}
    """
    merged_lineage = {}
    num_references = len(lineages)

    if num_references > 0:
        for lineage in lineages:
            for client_id, value in lineage.items():
                if client_id in merged_lineage:
                    merged_lineage[client_id] += value
                else:
                    merged_lineage[client_id] = value

        for client_id in merged_lineage:
            merged_lineage[client_id] /= num_references

    return merged_lineage


def compute_lineage(node, genesis):
    """
        Computes the lineage of a model while ensuring correct normalization.
        Being called after the model has added to dag as a tip.
        """
    # todo delete
    if node.id == 0:
        return {}
    # If model references is the Genesis
    if len(node.ref_nodes) == 1 and node.ref_nodes[0].id == genesis.id:
    # if node.ref_nodes == [genesis]:
        return {node.client_id: 1}
    reference_lineages = []
    for ref_node in node.ref_nodes:
        # print(f"node:{node.id}, c_id:{node.client_id}, ref_node:{ref_node.id}, ref_nodes.lineage:{ref_node.lineage}")
        reference_lineages.append(ref_node.lineage)
    merged_lineage = fusion(reference_lineages)
    # # Avoid the case for virtual global model GS calculation in tips selection alg.
    if node.id >= 0:
        merged_lineage[node.client_id] = merged_lineage.get(node.client_id, 0) + 1

    return merged_lineage

def compute_gs(node, observed_num_clients, a=1):
    """
    Computes the Generalization Score (M) for each DAG node.
    The number of nodes has presented in DAG up to this round is based on the observation of previous round
    The 0th round is special, we assume its observation is based on the 0th round's tips presence.
    """
    non_zero_clients = sum(1 for value in node.lineage.values() if value > 0)
    if len(node.lineage) > 1:
        std_lineage = np.std(list(node.lineage.values()))
    else:
        std_lineage = 0
    v = non_zero_clients / (observed_num_clients * (1 + a * std_lineage))
    return v
