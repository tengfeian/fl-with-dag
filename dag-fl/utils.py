import json
import torch

def write_logs(logfile, loss_list,accuracy_list, sim_name):
    # ensemble_log = args.log_file
    with open(logfile, mode="a") as file:
        file.write(sim_name+'\n')
        file.write(",".join(str(v) for v in accuracy_list))
        file.write('\n')
        file.write(",".join(str(v) for v in loss_list))
        file.write('\n')

def record_tips_info(file, rand_num_tips_all_rounds, new_tips_all_rounds, seed):
    with open(file, mode="a") as file:
        file.write("rand_num_tips_all_rounds with seed of "+str(seed) + ':\n')
        json.dump(rand_num_tips_all_rounds, file, indent=4)
        file.write('\n')
        file.write("tips_all_rounds with seed of " + str(seed) + ':\n')
        json.dump(new_tips_all_rounds, file, indent=4)
        file.write('\n')

# def record_tips_info(file, rand_num_tips_all_rounds, new_tips_all_rounds, seed):
#     with open(file, mode="a") as file:
#         file.write("rand_num_tips_all_rounds with seed of "+str(seed) + ':\n')
#         file.write(",".join(str(v) for v in rand_num_tips_all_rounds))
#         file.write('\n')
#
#         file.write("tips_all_rounds with seed of "+str(seed) + ':\n')
#         for tips in new_tips_all_rounds:
#             file.write('[')
#             file.write(",".join(str(v) for v in tips))
#             file.write(']\n')

def record_dag_info(dag, lineage_cache, gs_cache, seed, model_presence, model_birth, sim_name):
    with open("DAG.json", mode="a") as file:
        file.write("["+sim_name+"] DAG with seed of "+str(seed) + ':\n')
        json.dump(dag, file, indent=4)
        file.write('\n')
    if len(lineage_cache) > 0:
        with open("lineage_cache.json", mode="a") as file:
            file.write("["+sim_name+"] lineage_cache with seed of " + str(seed) + ':\n')
            json.dump(lineage_cache, file, indent=4)
            file.write('\n')
    if len(gs_cache) > 0:
        with open("GS_cache.json", mode="a") as file:
            file.write("["+sim_name+"] GS_cache with seed of " + str(seed) + ':\n')
            json.dump(gs_cache, file, indent=4)
            file.write('\n')
    with open("model_birth.json", mode="a") as file:
        file.write("["+sim_name+"] model_birth with seed of " + str(seed) + ':\n')
        json.dump(model_birth, file, indent=4)
        file.write('\n')
    with open("model_presence.json", mode="a") as file:
        file.write("["+sim_name+"] model_presence with seed of " + str(seed) + ':\n')
        json.dump(model_presence, file, indent=4)
        file.write('\n')


# def record_partition_metadata(metadata, dataset_name, seed):
#     with open("dataset_partitions.json", "a") as file:
#         file.write("split "+dataset_name+" with seed of "+str(seed)+ '\n')
#         json.dump(metadata, file)

# Function to check if parameters are identical
def models_equal(model_a, model_b):
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    # Ensure same keys
    if set(state_a.keys()) != set(state_b.keys()):
        return False

    # Compare each tensor
    for key in state_a:
        if not torch.equal(state_a[key], state_b[key]):
            print(f"Mismatch in parameter: {key}")  # Optional: for debugging
            return False

    return True