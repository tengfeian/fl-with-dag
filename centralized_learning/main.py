from run_centralized import run_centralised
import argparse
import torch
import json

parser = argparse.ArgumentParser(description='Centralized Learning Simulations')
parser.add_argument('--log_file', type=str, default='centralized_learning')
parser.add_argument('--net_name', type=str, default='CNN_CIFFAR10', choices=['CNN_MNIST', 'CNN_CIFFAR10'])
parser.add_argument('--dataset_name', type=str, default='cifar10', choices=['mnist', 'cifar10'])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == "__main__":
    args = parser.parse_args()
    loss_list, accuracy_list = run_centralised(args.net_name, args.dataset_name, args.batch_size, args.epochs, args.lr, args.momentum, DEVICE)
    log_file = args.log_file
    with open(f"{log_file}.json", "a") as f:
        json.dump({"loss": loss_list, "accuracy": accuracy_list}, f)
        f.write('\n')
