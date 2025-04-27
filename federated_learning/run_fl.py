from flwr.simulation import run_simulation
from client_app import gen_client_fn
from flwr.client import ClientApp
from server_app import gen_server_fn
from flwr.server import ServerApp
from datafactory import get_testloader

def run_fl(args, device):
    client_fn = gen_client_fn(args, device)
    testloader = get_testloader(args.dataset_name, args.partition_type, args.num_clients, args.batch_size)
    server_fn = gen_server_fn(args, testloader, device)

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)
    num_clients = args.num_clients

    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=num_clients
    )