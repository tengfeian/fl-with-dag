from run_fl import run_fl
import argparse
import torch
# from util import set_log

parser = argparse.ArgumentParser(description='Federated Learning Simulations')
parser.add_argument('--file_name', type=str, default='fl')
parser.add_argument('--net', type=str, default='CNN_CIFFAR10', choices=['CNN_MNIST', 'CNN_CIFFAR10'])
parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['mnist', 'cifar10'])

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--num_rounds', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--num_clients', type=int, default=10)
parser.add_argument('--partition_type', type=str, default="iid", choices=['iid', 'dirichlet'])
parser.add_argument('--min_available_clients', type=int, default=2)
parser.add_argument('--fraction_fit', type=float, default=1.0)
parser.add_argument('--fraction_evaluate', type=float, default=1.0)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    args = parser.parse_args()
    run_fl(args, DEVICE)
    # accuracies, losses = run_fl(args, DEVICE)
    # set_log(args.log_file)
    # ensemble_log = args.log_file
    # with open(ensemble_log, mode="a") as file:
    #     file.write(args.log_file+'\n')
    #     file.write(",".join(str(v) for v in accuracies))
    #     file.write('\n')
    #     file.write(",".join(str(v) for v in losses))
    #     file.write('\n')

