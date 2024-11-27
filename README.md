## Usage
1. config experiment info in "confg/exp.conf" file
```
[experiment_id]
context_size = 5
rounds = 30
hyp_rounds = 0
workload_shifts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,25,26,27,28,29,30]
is_retrain = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0]
queries_start = [0, 0, 0, 0, 0, 4,4,4,4,4,8,8,8,8,8,12,12,12,12,12,16,16,16,16,16,20,20,20,20,20,24,24,24,24,24,28,28,28,28,28,32,32,32,32,32]
queries_end = [20,20,20,20,20,24,24,24,24,24,28,28,28,28,28,32,32,32,32,32,36,36,36,36,36,40,40,40,40,40,44,44,44,44,44,48,48,48,48,48,52,52,52,52,52]
ta_runs = [1]
ta_workload = optimal
workload_file = /resources/workloads/job_static.json
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
components = ["MAB"]
mab_versions = ["simulation.sim_c3ucb_vrfc"]
workload_size = 33
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

```
python sim_run_experiment.py
```
