from dag_fl_alg import compute_lineage,compute_gs

class DAGNode:
    # def __init__(self, node_id, model, client_id, creation_time, ref_nodes, tips):
    def __init__(self, node_id, model, client_id, creation_time, ref_nodes):
        self.id = node_id
        self.model = model
        self.client_id = client_id
        self.creation_time = creation_time
        self.appearance_time = -1
        self.ref_nodes = ref_nodes
        self.lineage = None # {client_id: value}
        self.gs = None
        # self.tips_at_creation = tips # [tip DAGNode]
        self.cs = None # Choice Score is the GS of the virtual global model when this model is attached to the DAG

    def update_at_appearance(self, round, genesis, observed_num_clients):
        # To avoid unused tips being updated repeatedly in following rounds
        if self.appearance_time < 0:
            self.appearance_time = round
            self.set_lineage(genesis)
            self.set_gs(observed_num_clients)

    def set_lineage(self, genesis):
        self.lineage = compute_lineage(self, genesis)

    def set_gs(self, observed_num_clients):
        self.gs = compute_gs(self, observed_num_clients)

    def set_cs(self, cs):
        self.cs = cs




