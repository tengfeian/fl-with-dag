# IID, without tip selection by test accuracy
python main.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5


# non-IID, without tip selection by test accuracy
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5

# non-IID, Only for tip selection by test accuracy
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8457632145  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2


# IID, Only for tip selection by test accuracy
python main_tip_by_acc.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8457632145  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2


# non-IID, without tip selection by test accuracy
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5

# non-IID, Only for tip selection by test accuracy
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=12345486  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2


# non-IID, without tip selection by test accuracy
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5

# non-IID, Only for tip selection by test accuracy
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=64932040  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=9457284  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.1 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=1.0 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --seed=8273649  --num_clients=100 --dirichlet_alpha=0.5 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2


# IID, without tip selection by test accuracy
python main.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=5
python main.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=5

# IID, Only for tip selection by test accuracy
python main_tip_by_acc.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=12345486  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=64932040  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=9457284  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.25 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=1.0 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.7 --num_global_round=100 --num_tips_selected=2
python main_tip_by_acc.py --partition_type=iid --seed=8273649  --num_clients=100 --min_ratio_presence=0.5 --num_global_round=100 --num_tips_selected=2
