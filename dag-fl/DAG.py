from DAGNode import DAGNode
import copy

class DAG:
    def __init__(self):
        """
        model_to_references: {node_id: [ref_node_id]
        """
        self.genesis = None
        self.node_id_to_node = {} # {node_id: DAGNode}
        self.node_id_to_ref_ids = {} # represents the graph, {node_id: set(ref_node_id}
        self.tips = [] # [DAGNode]
        self.round_to_tips = {} # {round: [DAGNode]}
        # self.round_to_tip_nodes = {} # {round: [tip_node]}
        # self.round_to_tip_models = {} # {round: [tip_model]}
        # self.round_to_unused_tips = {} # {round: [DAGNode]}
        self.round_to_tip_gs = {} # {round: GS score for virtual global model}

    def init(self,genesis):
        self.genesis = genesis
        self.node_id_to_node[0] = genesis
        self.node_id_to_ref_ids[0] = []
        self.tips = [genesis]
        # self.tips = []
        self.round_to_tips[0] = [genesis]
        self.round_to_tip_gs[0] = 0

    def set_node_id_to_ref_ids(self, node, ref_nodes):
        self.node_id_to_ref_ids[node.id] = []
        if len(ref_nodes) == 0:
            self.node_id_to_ref_ids[node.id] = []
        else:
            for ref_node in ref_nodes:
                self.node_id_to_ref_ids[node.id].append(ref_node.id)

    def add_tips(self,round,clients, client_ids_current_round):
        # self.set_round_to_tips(round,clients,client_ids_current_round)
        for c_id in client_ids_current_round:
            c = clients[c_id]
            node = c.dag_node
            # node = copy.deepcopy(c.dag_node)
            ref_nodes = node.ref_nodes
            self.node_id_to_node[node.id] = node
            self.set_node_id_to_ref_ids(node, ref_nodes)

            # Remove previous tips which are parents of this new tip
            for ref_node in ref_nodes:
                if ref_node in self.tips:
                    self.tips.remove(ref_node)
                    # Clear the neural net of obsoleted node, to save memory
                    ref_node.model = None
            # Add new tip
            self.tips.append(node)

            # Record current DAG tips
        self.round_to_tips[round] = copy.copy(self.tips)

    # def print_tips(self):
    #     t_ids = []
    #     for t in self.tips:
    #         t_ids.append(t.id)
    #     print(f"[DAG.add-tips] current tips: {t_ids}")

    def set_round_to_tip_gs(self,round):
        """
        For virtual global model, the number of observed clients must be equal to the number of clients in its lineage
        """
        virtual_final_node_id = -1
        virtual_global_node = DAGNode(virtual_final_node_id, None, -1, round, self.tips)
        virtual_global_node.set_lineage(self.genesis)

        # todo, disputing, whether we should set all number of clients
        virtual_observed_num_clients = sum(1 for value in virtual_global_node.lineage.values() if value > 0)
        # todo delete
        # print(f"round: {round}, virtual_observed_num_clients: {virtual_observed_num_clients}")
        virtual_global_node.set_gs(virtual_observed_num_clients)
        self.round_to_tip_gs[round] = virtual_global_node.gs