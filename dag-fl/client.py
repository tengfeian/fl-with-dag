from learning import *
from aggregation import *

class Client:
    def __init__(self, client_id, local_model,num_local_epochs, device):
        """
        @type dag_node_history: list of DAGNode proposed by this client
        """
        self.id = client_id
        self.local_model = local_model # net refers to the AI neural network
        self.dag_node = None
        self.dag_node_history = []
        self.num_local_epochs = num_local_epochs
        self.trainloader = None
        self.testloader = None
        self.optimizer = None
        self.device = device

    def local_train(self):
        optim = get_optimizer(self.local_model)
        train(self.local_model, self.trainloader, optim, self.num_local_epochs, self.device)

    # todo Discuss: use training dataset to test tips
    def local_test(self, model):
        loss, accuracy = test(model, self.trainloader, self.device)
        return loss, accuracy

    def update_local_model(self, selected_tips):
        """

        @type selected_tips: list of DAGNode
        """
        self.dag_node_history.append(self.dag_node)
        selected_models = []
        for tip in selected_tips:
            selected_models.append(tip.model.to(self.device))
        self.local_model = fedAvg(selected_models)
