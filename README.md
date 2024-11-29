## Usage
1. config experiment info in "confg/exp.conf" file, there is an example in the conf
```
[tpc_h_static_refiner]
context_size = 5
reps = 1
rounds = 2
hyp_rounds = 0
workload_shifts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25,26,27,28,29,30]
is_retrain = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
queries_start = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
queries_end = [21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21]
ta_runs = [1]
ta_workload = optimal
workload_file = /resources/workloads/tpc_h_static_100_pg.json
config_shifts = [0]
config_start = [0]
config_end = [20]
max_memory = 25000
input_v1_square = 5
input_v2_square = 1
input_rho = 0.25
n = 2
input_alpha = 1
input_lambda = 0.5
time_weight = 5
memory_weight = 0
components = ["REFINER"]
workload_size = 22
budget = 4
```
config db info in "config/db.conf" file
```
[postgresql]
host =
database =
port =
user =
password =
```

2. run experiment
set experiment id in sim_run_experiment.py
```
exp_id_list = ["tpc_h_static_refiner"]
```
run sim_run_experiment.py
```
python sim_run_experiment.py
```
