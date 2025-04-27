"""dag-flwr-app: A Flower / PyTorch app."""

import torch
from flwr.common import Context
from typing import Callable

from flwr.client import NumPyClient
from flwr.common import Context
from nets import get_net
from learning import train, test, get_optimizer, get_weights, set_weights
from datafactory import get_dataloaders

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, local_epochs, device):
        self.net = net
        self.trainloader = trainloader
        # self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = device
        self.net.to(self.device)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        # loss, accuracy = test(self.net, self.valloader, self.device)
        loss, accuracy = test(self.net, self.trainloader, self.device)
        # return loss, len(self.valloader.dataset), {"accuracy": accuracy}
        return loss, len(self.trainloader.dataset), {"accuracy": accuracy}

def gen_client_fn(args, device) -> Callable[[Context], FlowerClient]:
    def client_fn(context: Context):
        net = get_net(args.net)
        local_epochs = args.local_epochs
        partition_id = context.node_config["partition-id"]
        # num_partitions = context.node_config["num-partitions"]
        # local_epochs = context.run_config["local-epochs"]
        # partition_id = args.partition_id
        # num_partitions = args.num_partitions
        batch_size = args.batch_size
        dataset_name = args.dataset_name
        partition_type = args.partition_type
        num_clients = args.num_clients
        # trainloader, valloader = get_dataloaders(partition_id, dataset_name, batch_size, partition_type, num_clients)
        trainloader = get_dataloaders(partition_id, dataset_name, batch_size, partition_type, num_clients)

        # Return Client instance
        # return FlowerClient(net, trainloader, valloader, local_epochs, device).to_client()
        return FlowerClient(net, trainloader, local_epochs, device).to_client()
    return client_fn
