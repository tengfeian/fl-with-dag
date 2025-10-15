import argparse
import torch
import random
from nets import get_net
from datafactory import split_dataset
from sim_dag_fl import run_dag_fl
from utils import write_logs,record_tips_info
from sim_central_rand_order import run_centralized_randomized_order
from sim_classic_seq import run_classic_sequential
from sim_traditional_fl import run_traditional_fl
from sim_dag_rand_tips import run_dag_rand_tips
from sim_dag_best_acc_tips import run_dag_best_acc_tips

parser = argparse.ArgumentParser(description='Federated Learning Simulations')
# parser.add_argument('--file_name', type=str, default='fl')
parser.add_argument('--log_file', type=str, default='log.log')
parser.add_argument('--record_file', type=str, default='records.log')
parser.add_argument('--net', type=str, default='MLP_MNIST', choices=['MLP_MNIST', 'CNN_CIFFAR10'])
parser.add_argument('--dataset_name', type=str, default='mnist', choices=['mnist', 'cifar10'])

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_local_epoch', type=int, default=1)
parser.add_argument('--num_global_round', type=int, default=2)
# parser.add_argument('--lr', type=float, default=0.01)
# parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--num_clients', type=int, default=100)
parser.add_argument('--partition_type', type=str, default="dirichlet", choices=['iid', 'dirichlet'])
parser.add_argument('--dirichlet_alpha', type=float, default=0.1)
parser.add_argument('--min_ratio_presence', type=float, default=0.7)
# Note: some seed with dirichlet_alpha=0.1 will cause some client has 0 samples, such as :
# 194753, 7133994, 302483932,
# ValueError: num_samples should be a positive integer value, but got num_samples=0
# Some safe seeds with dirichlet_a=[0.1,0.5,1.0] are [8457632145,12345486, 9457284, 8273649, 64932040, 54264622, 293994]
parser.add_argument('--seed', type=int, default=8457632145)
parser.add_argument('--num_tips_selected', type=int, default=2)

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
    rounds = args.num_global_round
    num_local_epoch = args.num_local_epoch
    min_ratio_presence = args.min_ratio_presence
    num_tips_selected = args.num_tips_selected
    seed = args.seed

    log_file = args.log_file
    record_file = args.record_file
    # DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    initial_net = get_net(args.net)
    random.seed(seed)
    torch.manual_seed(seed)
    # np.random.seed(seed)
    fds = split_dataset(dataset_name, partition_type, num_clients, dirichlet_alpha, seed)
    # record_partition_metadata(fds.partion_metadata, dataset_name, seed)

    """
        All the asynchronous scenarios of DAG-FL and FL share the same random distribution in tips.
    """
    min_num_new_tips = int(min_ratio_presence * num_clients)
    clients_list = list(range(num_clients))
    all_round_to_clients = []
    rand_num_tips_all_rounds = []
    for e in list(range(rounds)):
        if e == 0:
            num_tips_current_round = num_clients
            new_tips_current_round = list(range(num_clients))
        else:
            num_tips_current_round = random.randint(min_num_new_tips, num_clients)
            new_tips_current_round = random.sample(clients_list, num_tips_current_round)
        rand_num_tips_all_rounds.append(num_tips_current_round)
        all_round_to_clients.append(new_tips_current_round)
    record_tips_info(record_file, rand_num_tips_all_rounds, all_round_to_clients, seed)

    # # Run traditional synchronous FL, only for sync environment without drop-offs
    # if min_ratio_presence == 1.0 :
    #     label = "[Sync_FL]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    #     loss_list, accuracy_list = run_traditional_fl(fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE)
    #     write_logs(log_file,loss_list, accuracy_list, label)

    # # Run new DAG-FL alg.
    # label = "[DAGFL_GS_alg.]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    # loss_list, accuracy_list = run_dag_fl(fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients, label)
    # write_logs(log_file,loss_list, accuracy_list, label)

    # Run random tip selection DAG-FL
    label = "[DAGFL_RandTips_alg.]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    loss_list, accuracy_list = run_dag_rand_tips(num_tips_selected, fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients, label)
    write_logs(log_file,loss_list, accuracy_list, label)

    # # Run traditional Asynchronous FL
    # if min_ratio_presence < 1.0:
    #     label = "[Async_FL]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    #     loss_list, accuracy_list = run_traditional_fl(fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients)
    #     write_logs(log_file,loss_list, accuracy_list, label)
    #
    # # Run classic fully centralized sequential training
    # label = "[Central_Sequential]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    # loss_list, accuracy_list = run_classic_sequential(fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients)
    # write_logs(log_file,loss_list, accuracy_list, label)
    #
    # # Run centralized training with randomized order
    # label = "[Central_RandOrder]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    # loss_list, accuracy_list = run_centralized_randomized_order(fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients)
    # write_logs(log_file,loss_list, accuracy_list, label)

    # # Run DAG-FL with tip selection based on test accuracy
    # label = "[DAGFL_BestAccTips_alg.]-"+args.partition_type+"-num_local_epoch_"+str(args.num_local_epoch)+"-"+args.dataset_name+"-num_clients_"+str(args.num_clients)+"-min_ratio_presence_"+str(min_ratio_presence)+"-dirichlet_alpha_"+str(dirichlet_alpha)+"-seed_"+str(seed)
    # loss_list, accuracy_list = run_dag_best_acc_tips(num_tips_selected, fds, initial_net, dataset_name, batch_size, rounds, num_local_epoch, num_clients, DEVICE, all_round_to_clients, label)
    # write_logs(log_file,loss_list, accuracy_list, label)