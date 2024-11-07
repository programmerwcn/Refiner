import pickle
from importlib import reload
import logging

from pandas import DataFrame

#Add '../' to path to import from parent directory
# import sys
# import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# relative_path = os.path.join(current_dir, '../')
# sys.path.append(relative_path)

import constants
# from bandits.experiment_report import ExpReport
# from database.config_test_run import ConfigRunner
from shared import configs_v2 as configs, helper

# Define Experiment ID list that we need to run
exp_id_list = ["tpc_h_session_10_MAB_rfc_epsilon0.5"]
# Comparing components
MAB = constants.COMPONENT_MAB in configs.components
ACC_UCB = constants.COMPONENT_ACC_UCB in configs.components

# Generate form saved reports
FROM_FILE = False
SEPARATE_EXPERIMENTS = True
PLOT_LOG_Y = False
PLOT_MEASURE = (constants.MEASURE_BATCH_TIME, constants.MEASURE_QUERY_EXECUTION_COST,
                constants.MEASURE_INDEX_CREATION_COST)
UNIFORM = False

exp_report_list = []

for i in range(len(exp_id_list)):
    
    if SEPARATE_EXPERIMENTS:
        exp_report_list = []
    experiment_folder_path = helper.get_experiment_folder_path(exp_id_list[i])
    helper.change_experiment(exp_id_list[i])
    reload(configs)
    reload(logging)

     # configuring the logger
    if not FROM_FILE:
        logging.basicConfig(
            filename=experiment_folder_path + configs.experiment_id + '.log',
            filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')
        logging.getLogger().setLevel(constants.LOGGING_LEVEL)

    if FROM_FILE:
        with open(experiment_folder_path + "reports.pickle", "rb") as f:
            exp_report_list = exp_report_list + pickle.load(f)
    else:
        print("Currently running: ", exp_id_list[i])
        # Running MAB
        if MAB:
            Simulators = {}
            for mab_version in configs.mab_versions:
                Simulators[mab_version] = (getattr(__import__(mab_version, fromlist=['Simulator']), 'Simulator'))
            for version, Simulator in Simulators.items():
                version_number = version.split("_v", 1)[1]
                exp_report_mab = ExpReport(configs.experiment_id,
                                           constants.COMPONENT_MAB + version_number + exp_id_list[i], configs.reps,
                                           configs.rounds)
                for r in range(configs.reps):
                    simulator = Simulator()
                    results, total_workload_time = simulator.run()
                    temp = DataFrame(results, columns=[constants.DF_COL_BATCH, constants.DF_COL_MEASURE_NAME,
                                                       constants.DF_COL_MEASURE_VALUE])
                    temp._append([-1, constants.MEASURE_TOTAL_WORKLOAD_TIME, total_workload_time])
                    temp[constants.DF_COL_REP] = r
                    exp_report_mab.add_data_list(temp)
                exp_report_list.append(exp_report_mab)
        # Running ACC UCB
        if ACC_UCB:
            from sim_acc_ucb import Simulator as ACCUCBSimulator 
            exp_report_acc_ucb = ExpReport(configs.experiment_id, constants.COMPONENT_ACC_UCB, configs.reps, configs.rounds)
            for r in range(configs.reps):
                simulator = ACCUCBSimulator()
                results, total_workload_time = simulator.run() 
                temp = DataFrame(results, columns=[constants.DF_COL_BATCH, constants.DF_COL_MEASURE_NAME,
                                                       constants.DF_COL_MEASURE_VALUE])
                temp._append([-1, constants.MEASURE_TOTAL_WORKLOAD_TIME, total_workload_time])
                temp[constants.DF_COL_REP] = r
                exp_report_acc_ucb.add_data_list(temp)
            exp_report_list.append(exp_report_acc_ucb)

        # Save results
        with open(experiment_folder_path + "reports.pickle", "wb") as f:
            pickle.dump(exp_report_list, f)

# plot line graphs
if not SEPARATE_EXPERIMENTS:
    helper.plot_exp_report(configs.experiment_id, exp_report_list, PLOT_MEASURE, PLOT_LOG_Y)
    helper.create_comparison_tables(configs.experiment_id, exp_report_list)
        
