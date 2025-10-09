import argparse
import torch
import random
from nets import get_net
from datafactory import split_dataset
from sim_dag_fl import run_dag_fl
from sim_baseline_dagfl_i import run_baseline_dagfl_i
from utils import write_logs,record_tips_info
from sim_central_rand_order import run_centralized_randomized_order
from sim_classic_seq import run_classic_sequential
from sim_traditional_fl import run_traditional_fl

parser = argparse.ArgumentParser(description='Federated Learning Simulations')
# parser.add_argument('--file_name', type=str, default='fl')
parser.add_argument('--log_file', type=str, default='logs.log')
parser.add_argument('--record_file', type=str, default='records.log')
parser.add_argument('--net', type=str, default='MLP_MNIST', choices=['MLP_MNIST', 'CNN_CIFFAR10'])
parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'cifar10'])

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--local_epochs', type=int, default=1)
parser.add_argument('--global_epochs', type=int, default=100)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--num_clients', type=int, default=20)
parser.add_argument('--partition_type', type=str, default="dirichlet", choices=['iid', 'dirichlet'])
parser.add_argument('--dirichlet_alpha', type=float, default=0.1)
parser.add_argument('--min_ratio_presence', type=float, default=0.7)
parser.add_argument('--seed', type=int, default=8457632145)

# parser.add_argument('--min_available_clients', type=int, default=16)
# parser.add_argument('--fraction_fit', type=float, default=1.0)
# parser.add_argument('--fraction_evaluate', type=float, default=1.0)

if __name__ == "__main__":
    args = parser.parse_args()
    dataset_name = args.dataset_name
    partition_type = args.partition_type
    num_clients = args.num_clients
    dirichlet_alpha = args.dirichlet_alpha
    batch_size = args.batch_size
    epochs = args.global_epochs
    local_epochs = args.local_epochs
    min_ratio_presence = args.min_ratio_presence
    seed = args.seed

    log_file = args.log_file
    record_file = args.record_file
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    initial_net = get_net(args.net)
    random.seed(seed)
    torch.manual_seed(seed)
    # np.random.seed(seed)
    fds = split_dataset(dataset_name, partition_type, num_clients, dirichlet_alpha, seed)
    # record_partition_metadata(fds.partion_metadata, dataset_name, seed)

    # Run classic fully centralized sequential training
    loss_list, accuracy_list = run_classic_sequential(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE)
    write_logs(log_file,loss_list, accuracy_list, "[classic_sequential]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))

    # Run centralized training with randomized order
    loss_list, accuracy_list = run_centralized_randomized_order(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE)
    write_logs(log_file,loss_list, accuracy_list, "[centralized_randomized_order]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))

    # Run traditional synchronous FL
    loss_list, accuracy_list = run_traditional_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE)
    write_logs(log_file,loss_list, accuracy_list, "[traditional synchronous FL]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))

    """
    All the asynchronous scenarios of DAG-FL and FL share the same random distribution in tips.
    """
    min_num_new_tips = int(min_ratio_presence*num_clients)
    clients_list = list(range(num_clients))
    new_tips_all_rounds = []
    rand_num_tips_all_rounds = []
    for e in list(range(epochs)):
        if e==0:
            num_tips_current_round= 10
            new_tips_current_round = list(range(num_clients))
        else:
            num_tips_current_round = random.randint(min_num_new_tips, num_clients)
            new_tips_current_round = random.sample(clients_list, num_tips_current_round)
        rand_num_tips_all_rounds.append(num_tips_current_round)
        new_tips_all_rounds.append(new_tips_current_round)
    record_tips_info(record_file, rand_num_tips_all_rounds, new_tips_all_rounds, seed)

    # Run new DAG-FL alg.
    loss_list, accuracy_list = run_dag_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE, new_tips_all_rounds, seed)
    write_logs(log_file,loss_list, accuracy_list, "[DAG-FL alg.]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))

    # Run traditional Asynchronous FL
    loss_list, accuracy_list = run_traditional_fl(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE, new_tips_all_rounds)
    write_logs(log_file,loss_list, accuracy_list, "[traditional Asynchronous FL]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))

    # # Run baseline DAG-FL A
    loss_list, accuracy_list = run_baseline_dagfl_i(fds, initial_net, dataset_name, batch_size, epochs, local_epochs, num_clients, DEVICE, new_tips_all_rounds, seed)
    write_logs(log_file,loss_list, accuracy_list, "[Baseline DAG-FL I]-"+args.partition_type+"-local_epochs_"+str(args.local_epochs)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed))


