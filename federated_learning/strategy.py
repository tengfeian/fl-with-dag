from flwr.server.strategy import FedAvg
from flwr.common import Parameters
import json
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from learning import test

class FedAvgCustom(FedAvg):
    def __init__(self, file_name: str, num_rounds: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file_name = file_name
        self.num_rounds = num_rounds
        self.loss_list = []
        self.accuracy_list = []
        # self.metrics_list = []

    # def _make_plot(self):
    #     """Makes a plot with the results recorded"""
    #     round = list(range(1, len(self.loss_list) + 1))
    #     acc = [100.0 * metrics["accuracy"] for metrics in self.metrics_list]
    #     plt.plot(round, acc)
    #     plt.grid()
    #     plt.ylabel("Accuracy (%)")
    #     plt.xlabel("Round")

    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate model parameters using an evaluation function."""
        # loss, metrics = super().evaluate(server_round, parameters)
        loss, accuracy = super().evaluate(server_round, parameters)
        # Record results
        self.loss_list.append(loss)
        self.accuracy_list.append(accuracy)
        print(f"Global accuracy of each round: {self.accuracy_list}")
        # If last round, save results and make a plot
        if server_round == self.num_rounds:
            # Save to CSV
            with open(f"{self.file_name}.json", "a") as f:
                json.dump({"loss": self.loss_list, "accuracy": self.accuracy_list}, f)
                f.write('\n')
            # Generate plot
            # self._make_plot()


def get_evaluate_fn(net, testloader, device):
    """Return a function that can be called to do global evaluation."""

    def evaluate_fn(server_round: int, parameters, config):
        """Evaluate global model on the whole test set."""

        model = net.to(device)

        # set parameters to the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # call test (evaluate model as in centralised setting)
        loss, accuracy = test(model, testloader, device)
        return loss, accuracy
        # return loss, {"accuracy": accuracy}

    return evaluate_fn