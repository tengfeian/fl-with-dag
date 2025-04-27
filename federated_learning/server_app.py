"""dag-flwr-app: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from strategy import FedAvgCustom, get_evaluate_fn
from nets import get_net
from learning import get_weights


def gen_server_fn(args, testloader, device):

    def server_fn(context: Context):
        # Read from config
        # num_rounds = context.run_config["num-server-rounds"]
        # fraction_fit = context.run_config["fraction-fit"]

        # Initialize model parameters
        net = get_net(args.net)
        ndarrays = get_weights(net)
        parameters = ndarrays_to_parameters(ndarrays)

        # Define strategy
        strategy = FedAvgCustom(
            file_name=args.file_name,
            num_rounds=args.num_rounds,
            fraction_fit=args.fraction_fit,
            fraction_evaluate=args.fraction_evaluate,
            min_available_clients=args.min_available_clients,
            initial_parameters=parameters,
            evaluate_fn=get_evaluate_fn(net,testloader,device)
        )
        config = ServerConfig(num_rounds=args.num_rounds)

        return ServerAppComponents(strategy=strategy, config=config)
    return server_fn
