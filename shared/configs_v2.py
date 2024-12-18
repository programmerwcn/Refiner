import configparser
import json

import constants as constants

# Reading the configuration for given experiment ID
exp_config = configparser.ConfigParser()
exp_config.read(f"{constants.ROOT_DIR}/{constants.EXPERIMENT_CONFIG}")

# experiment id for the current run
experiment_id = exp_config["general"]["run_experiment"]

# information about experiment
reps = int(exp_config[experiment_id]["reps"])
rounds = int(exp_config[experiment_id]["rounds"])
hyp_rounds = int(exp_config[experiment_id]["hyp_rounds"])
workload_shifts = json.loads(exp_config[experiment_id]["workload_shifts"])
queries_start_list = json.loads(exp_config[experiment_id]["queries_start"])
queries_end_list = json.loads(exp_config[experiment_id]["queries_end"])
config_shifts = json.loads(exp_config[experiment_id]["config_shifts"])
config_start_list = json.loads(exp_config[experiment_id]["config_start"])
config_end_list = json.loads(exp_config[experiment_id]["config_end"])
ta_runs = json.loads(exp_config[experiment_id]["ta_runs"])
ta_workload = str(exp_config[experiment_id]["ta_workload"])
workload_file = str(exp_config[experiment_id]["workload_file"])
components = json.loads(exp_config[experiment_id]["components"])
if experiment_id.find('MAB') != -1:
    mab_versions = json.loads(exp_config[experiment_id]["mab_versions"])
if "budget" in exp_config[experiment_id]:
    budget = int(exp_config[experiment_id]["budget"])
else:
    budget = 5
if "embedding_model" in exp_config[experiment_id]:
    embedding_model = str(exp_config[experiment_id]["embedding_model"])
else:
    embedding_model = None
if "max_workload_embedding" in exp_config[experiment_id]:
    max_workload_embedding = int(exp_config[experiment_id]["max_workload_embedding"])
else:
    max_workload_embedding = 10
if "alpha" in exp_config[experiment_id]:
    alpha = float(exp_config[experiment_id]["alpha"])
else:
    alpha = 0.001
if "epsilon" in exp_config[experiment_id]:
    epsilon = float(exp_config[experiment_id]["epsilon"])
else:
    epsilon = 0.5
if "sample_rate" in exp_config[experiment_id]:
    sample_rate = float(exp_config[experiment_id]["sample_rate"])
else:
    sample_rate = 0.1
# db_name = exp_config[experiment_id]['db_name']
# constraints
# constraint = exp_config[experiment_id]["constraint"]
max_memory = float(exp_config[experiment_id]["max_memory"])
# max_count = int(exp_config[experiment_id]["max_count"])

# hyper parameters
input_alpha = float(exp_config[experiment_id]["input_alpha"])
input_lambda = float(exp_config[experiment_id]["input_lambda"])


if experiment_id.find('ACC_UCB') != -1:
    input_v1_square = float(exp_config[experiment_id]['input_v1_square'])
    input_v2_square = float(exp_config[experiment_id]['input_v2_square'])
    N = int(exp_config[experiment_id]['N'])
    input_rho = float(exp_config[experiment_id]['input_rho'])
    context_size = exp_config[experiment_id]['context_size']

# if experiment_id.find('MAB') != -1:
#     workload_size = int(exp_config[experiment_id]['workload_size'])

if experiment_id.find('refiner') != -1 or experiment_id.find('sgd') != -1:
    is_retrain = json.loads(exp_config[experiment_id]['is_retrain'])
    workload_size = int(exp_config[experiment_id]['workload_size'])




   
