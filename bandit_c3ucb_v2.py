import logging
from abc import abstractmethod

import numpy

import constants

import database.sql_helper_v2 as sql_helper

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from river import tree
from river import metrics
# from river import ensemble
# from river import tree
# from river import metrics
# from river import synth
from scipy.stats import entropy

from functools import cmp_to_key

import random

import time
class C3UCBBaseBandit:

    def __init__(self, context_size, hyper_alpha, hyper_lambda, oracle):
        self.arms = []
        self.alpha_original = hyper_alpha
        self.hyper_alpha = hyper_alpha
        self.hyper_lambda = hyper_lambda        # lambda in C2CUB
        self.v = hyper_lambda * numpy.identity(context_size)    # identity matrix of n*n
        self.b = numpy.zeros((context_size, 1))  # [0, 0, ..., 0]T (column matrix) size = number of arms
        self.oracle = oracle
        self.context_vectors = []
        self.upper_bounds = []
        self.context_size = context_size
        self.clf = RandomForestClassifier()
        self.aclf = tree.HoeffdingTreeClassifier()
        self.training_set = {}
        self.training_data = {}
        self.context_dict = {}
        self.workload = []
        self.historical_workload_vectors = []
        self.cycle_workload_dict = {}
        self.is_retrain = []
        self.sgd_clf = SGDClassifier(loss='log', penalty='l2', max_iter=5000, tol=1e-3, learning_rate='optimal', eta0=0.01,random_state=42)
        self.classifier = "VFDT"
        self.model = None
        self.historical_arms = {}
        self.context_arm_dict = {}
        self.candidate_input_data = []
        self.retrain_point = 0
        self.alpha = 0.001
        self.epsilon = 0.5
        self.sample_rate = 0.2

    @abstractmethod
    def select_arm(self, context_vectors, current_round):
        pass

    @abstractmethod
    def update(self, played_arms, reward, index_use):
        pass


class C3UCB(C3UCBBaseBandit):

    def select_arm(self, context_vectors, current_round):
        pass

    def select_arm_v2(self, context_vectors, current_round, budget):
        """
        This method implements the C2UCB algorithm
        :param context_vectors: context vector for this round
        :param current_round: current round number
        :return: selected set of arms
        """
        # calculate the weight vector: v^-1 * b
        v_inverse = numpy.linalg.inv(self.v)
        weight_vector = v_inverse @ self.b
        logging.info(f"================================\n{weight_vector.transpose().tolist()[0]}")
        self.context_vectors = context_vectors

        # find the upper bound for every arm
        for i in range(len(self.arms)):
            creation_cost = weight_vector[1] * self.context_vectors[i][1]
            average_reward = numpy.ndarray.item(weight_vector.transpose() @ self.context_vectors[i]) - creation_cost
            temp_upper_bound = average_reward + self.hyper_alpha * numpy.sqrt(
                numpy.ndarray.item(self.context_vectors[i].transpose() @ v_inverse @ self.context_vectors[i]))
            temp_upper_bound = temp_upper_bound + (creation_cost/constants.CREATION_COST_REDUCTION_FACTOR)
            self.upper_bounds.append(temp_upper_bound)

        logging.debug(self.upper_bounds)
        self.hyper_alpha = self.hyper_alpha / constants.ALPHA_REDUCTION_RATE
        return self.oracle.get_super_arm(self.upper_bounds, self.context_vectors, self.arms, budget)
    
    def hyp_2_model_reward(self):
        # Get the sorting number of hyp_benefit of each arm, and store them in the arm object
        arms_sorted_by_hyp_benefit = sorted(self.arms, key=lambda x: x.hyp_benefit, reverse=False)
        current_max = 0
        for i in range(len(arms_sorted_by_hyp_benefit)):
            if i > 0 and (arms_sorted_by_hyp_benefit[i].hyp_benefit - current_max) < 0.01:
                arms_sorted_by_hyp_benefit[i].hyp_benefit = arms_sorted_by_hyp_benefit[i-1].hyp_benefit
            else:
                current_max = arms_sorted_by_hyp_benefit[i].hyp_benefit
                arms_sorted_by_hyp_benefit[i].hyp_benefit = i
        # Get the sorting number of upper_bound of each arm
        arms_sorted_by_upper_bound = sorted(self.arms, key=lambda x: x.upper_bound, reverse=False)
        current_max = 0
        for i in range(len(arms_sorted_by_upper_bound)):
            if i > 0 and (arms_sorted_by_upper_bound[i].upper_bound - current_max) < 0.01:
                arms_sorted_by_upper_bound[i].upper_bound = arms_sorted_by_upper_bound[i-1].upper_bound
            else:
                current_max = arms_sorted_by_upper_bound[i].upper_bound
                arms_sorted_by_upper_bound[i].upper_bound = i

    def reward_2_class(self, rewards, class_num):
        bins = numpy.linspace(min(rewards), max(rewards), class_num + 1)
    
        # 分配区间
        bin_indices = numpy.digitize(rewards, bins) - 1  # bins返回的是1-based索引，所以需要减1
        
        # 修正边界值
        bin_indices[bin_indices == class_num] = class_num - 1

        return bin_indices





    def select_arm_v3(self, context_vectors, current_round):
        """
        This method implements the C2UCB algorithm
        :param context_vectors: context vector for this round
        :param current_round: current round number
        :return: selected set of arms
        """
        # calculate the weight vector: v^-1 * b
        v_inverse = numpy.linalg.inv(self.v)
        weight_vector = v_inverse @ self.b
        logging.info(f"================================\n{weight_vector.transpose().tolist()[0]}")
        self.context_vectors = context_vectors

        # find the upper bound for every arm
        for i in range(len(self.arms)):
            creation_cost = weight_vector[1] * self.context_vectors[i][1]
            average_reward = numpy.ndarray.item(weight_vector.transpose() @ self.context_vectors[i]) - creation_cost
            confidence_interval = self.hyper_alpha * numpy.sqrt(
                numpy.ndarray.item(self.context_vectors[i].transpose() @ v_inverse @ self.context_vectors[i]))
            temp_upper_bound = average_reward + confidence_interval
            temp_upper_bound = temp_upper_bound + (creation_cost/constants.CREATION_COST_REDUCTION_FACTOR)
            self.arms[i].upper_bound = temp_upper_bound[0]

        class_labels = self.reward_2_class([arm.upper_bound for arm in self.arms], 10)
        hyp_class_labels = self.reward_2_class([arm.hyp_benefit for arm in self.arms], 10)

        # Compute the combined reward of model estimation and hyp estimation
        for i in range(len(self.arms)):
            confidence_interval = self.hyper_alpha * numpy.sqrt(
                numpy.ndarray.item(self.context_vectors[i].transpose() @ v_inverse @ self.context_vectors[i]))
            alpha = confidence_interval / (confidence_interval + 1)
            # dynamic_factor = beta * self.arms[i].hyp_benefit / (confidence_interval + epsilon) + (1 - beta)
            combined_reward = alpha * hyp_class_labels[i] + (1 - alpha) * class_labels[i]
            logging.info(f"Combined reward for {self.arms[i].index_name} is {combined_reward}, mab reward is {self.arms[i].upper_bound} with label {class_labels[i]}, hyp_benefit is {self.arms[i].hyp_benefit} with label {hyp_class_labels[i]}, confidence interval is {confidence_interval}")
            self.upper_bounds.append(combined_reward)

        logging.debug(self.upper_bounds)
        self.hyper_alpha = self.hyper_alpha / constants.ALPHA_REDUCTION_RATE
        return self.oracle.get_super_arm(self.upper_bounds, self.context_vectors, self.arms)
    
    def select_arm_test(self):
        for i in range(len(self.arms)):
            if self.arms[i].index_name == "ixn_lineitem_53_54_49include_51_50":
                return [i]
            # self.arms[i].upper_bound = self.arms[i].hyp_benefit
    # def select_arm_v5(self, context_vectors, current_round):
    #     """
    #     Get the combined reward of C2UCB algorithm & offline advisor by classification.
    #     """



    def comp_context_vectors(self, item_1, item_2):
        input_vector = numpy.concatenate((item_1[1][0] - item_2[1][0], item_1[1][1]))
        if self.classifier == "RandomForest":
            reward = self.clf.predict([input_vector])[0]
        elif self.classifier == "SGD":
            reward = self.sgd_clf.predict([input_vector])[0]
        elif self.classifier == "VFDT":
            input_vector_dict = {f'feature_{i}': value for i, value in enumerate(input_vector)}
            reward = self.aclf.predict_one(input_vector_dict)
        # if self.arms[item_1[0]].index_name == "ix_catalog_sales_3" or self.arms[item_2[0]].index_name == "ix_catalog_sales_3":
        #     logging.info(f"Reward for {self.arms[item_1[0]].index_name} and {self.arms[item_2[0]].index_name} is {reward} with input_vector{input_vector} ")
        return reward
    

    def uncertainty_sampling(self, candidate_inputs, n_samples):
        """
        Perform uncertainty sampling to select the n_samples most uncertain points from X_pool.
        """
        if self.classifier == "RandomForest":
            # Predict probabilities on the unlabeled data
            probs = self.clf.predict_proba(candidate_inputs)
            
            # Calculate uncertainty: use margin sampling (difference between the top two class probabilities)
            if probs.shape[1] == 2:
                # Binary classification case
                uncertainty = 1 - numpy.abs(probs[:, 0] - probs[:, 1])
            else:
                # Multiclass classification case
                sorted_probs = numpy.sort(probs, axis=1)
                uncertainty = sorted_probs[:, -1] - sorted_probs[:, -2]

            # Get the indices of the n_samples most uncertain points
            uncertain_indices = numpy.argsort(uncertainty)[-n_samples:]

            return uncertain_indices
        elif self.classifier == "VFDT":
            uncertainties = []
            for x in candidate_inputs:
                input_vector_dict = {f'feature_{i}': value for i, value in enumerate(x)}
                probs = list(self.aclf.predict_proba_one(input_vector_dict).values())
                uncertainties.append(entropy(probs))
            uncertain_indices = numpy.argsort(uncertainties)[-n_samples:]
            return uncertain_indices
       
                # reward = self.aclf.predict_one(input_vector_dict)

    
    
    @staticmethod
    def epsilon_greedy_selection(sorted_list, epsilon):
        """
        Select an item from a sorted list using the epsilon-greedy algorithm.

        Parameters:
        sorted_list (list): The sorted list of items.
        epsilon (float): The probability of selecting a random item.

        Returns:
        selected_item: The selected item from the list.
        """
        # Generate a random number
        r = random.random()

        # Exploit: Select the best item with probability (1 - epsilon)
        if r > epsilon:
            selected_item = sorted_list[0]
        # Explore: Select a random item with probability epsilon
        else:
            selected_item = random.choice(sorted_list)
        
        return selected_item

    def predict_two_indexes(self, index_name_1, index_name_2):
        context_diff = self.context_dict[index_name_1] - self.context_dict[index_name_2]
        input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[-1]))
        if self.classifier == "RandomForest":
            reward = self.clf.predict([input_vector])
        return reward[0]


    def select_arm_v4(self, context_vectors, current_round, workload_vector, budget):
        """
        Select an arm based on context vectors, current round, and workload vector.
        Arm value estimation: the given classifier

        Parameters:
        context_vectors (list): The list of context vectors.
        current_round (int): The current round number.
        workload_vector (list): The workload vector.

        Returns:
        chosen_arms: The list of chosen arms.
        """
        context_vectors_array = numpy.array(context_vectors)
        flattened_context_vectors = context_vectors_array.reshape(context_vectors_array.shape[0], -1)
        workload_vector = numpy.array(workload_vector).reshape(1, -1)[0]
        
        # Initialize: train clf with what-if calls
        if self.is_retrain[current_round]:
            self.retrain_point = current_round
            new_training_data = self.update_training_data_hyp(current_round)
            self.retrain_clf(new_training_data)
            # if classifier == "RandomForest":
            #     self.retrain_clf(self.training_data, "RandomForest")
            # elif classifier == "SGD":
            #     self.retrain_clf(self.training_data, "SGD")
            # self.retrain_clf()

        context_dict = {}
        for i in range(len(self.arms)):
            context_dict[i] = (flattened_context_vectors[i], workload_vector)

        arm_dict = {}
        for i in range(len(self.arms)):
            arm_dict[i] = self.arms[i]
        used_memory = 0
        chosen_arms = []
        table_count = {}
        max_count = budget

        # if current_round == 0:
        #     for i in range(len(self.arms)):
        #         if self.arms[i].index_name == "ix_catalog_sales_3" or self.arms[i].index_name == "ix_item_13_0_16":
        #             chosen_arms.append(i)
        #     return chosen_arms
        epsilon = 0.5 / ((current_round+1-self.retrain_point) ** self.epsilon)
        logging.info(f"Epsilon is {epsilon}")   
        while len(context_dict) > 1 and max_count > 0:
            sorted_context = sorted(context_dict.items(), key=cmp_to_key(self.comp_context_vectors), reverse=True)
            logging.info(f"Top candidate index:[{[self.arms[item[0]].index_name for item in sorted_context]}")
            selected_arm_id = self.epsilon_greedy_selection(sorted_context, epsilon)[0]
            if self.arms[selected_arm_id].memory < self.oracle.max_memory - used_memory:
                chosen_arms.append(selected_arm_id)
                used_memory += self.arms[selected_arm_id].memory
                if self.arms[selected_arm_id].table_name in table_count:
                    table_count[self.arms[selected_arm_id].table_name] += 1
                else:
                    table_count[self.arms[selected_arm_id].table_name] = 1
                # remove arms that are similar to max_ucb_arm_id
                # arm_class_dict = self.removed_covered_queries(arm_class_dict, selected_arm_id, bandit_arms)
                arm_dict = self.oracle.removed_covered_tables(arm_dict, selected_arm_id, self.arms, table_count)
                arm_dict = self.oracle.removed_covered_clusters(arm_dict, selected_arm_id, self.arms)
                arm_dict = self.oracle.removed_covered_queries_v2(arm_dict, selected_arm_id, self.arms)
                arm_dict = self.oracle.removed_covered_v2(arm_dict, selected_arm_id, self.arms,
                                                       self.oracle.max_memory - used_memory)
                arm_dict = self.oracle.removed_same_prefix(arm_dict, selected_arm_id, self.arms, 1)
            else:
                arm_dict.pop(selected_arm_id)
            max_count -= 1
            # Update contexts
            updated_context_dict = {}
            for arm_id, arm in arm_dict.items():
                updated_context_dict[arm_id] = context_dict[arm_id]
            context_dict = updated_context_dict
        return chosen_arms
    
    def select_arm_v5(self, context_vectors, current_round, workload_vector):
        """
        Select an arm based on context vectors, current round, and workload vector.
        Arm value estimation: the given classifier
        Model training: uncertainty sampling

        Parameters:
        context_vectors (list): The list of context vectors.
        current_round (int): The current round number.
        workload_vector (list): The workload vector.

        Returns:
        chosen_arms: The list of chosen arms.
        """
        context_vectors_array = numpy.array(context_vectors)
        flattened_context_vectors = context_vectors_array.reshape(context_vectors_array.shape[0], -1)
        workload_vector = numpy.array(workload_vector).reshape(1, -1)[0]
        
        # Initialize: train clf with what-if calls
        if self.is_retrain[current_round]:
            self.initialize_vfdt(current_round)
            # Evaluate the accurracy of the clf
            # input_vectors = []
            # output_labels = []
            
            # new_training_data = self.update_training_data_hyp(current_round)
            # self.retrain_clf(new_training_data)
            # if classifier == "RandomForest":
            #     self.retrain_clf(self.training_data, "RandomForest")
            # elif classifier == "SGD":
            #     self.retrain_clf(self.training_data, "SGD")
            # self.retrain_clf()

        context_dict = {}
        for i in range(len(self.arms)):
            context_dict[i] = (flattened_context_vectors[i], workload_vector)

        arm_dict = {}
        for i in range(len(self.arms)):
            arm_dict[i] = self.arms[i]
        used_memory = 0
        chosen_arms = []
        table_count = {}
        max_count = 4

        # if current_round == 0:
        #     for i in range(len(self.arms)):
        #         if self.arms[i].index_name == "ix_catalog_sales_3" or self.arms[i].index_name == "ix_item_13_0_16":
        #             chosen_arms.append(i)
        #     return chosen_arms
                
        while len(context_dict) > 1 and max_count > 0:
            sorted_context = sorted(context_dict.items(), key=cmp_to_key(self.comp_context_vectors), reverse=True)
            logging.info(f"Top candidate index:[{[self.arms[item[0]].index_name for item in sorted_context]}")
            selected_arm_id = self.epsilon_greedy_selection(sorted_context, 0.2)[0]
            if self.arms[selected_arm_id].memory < self.oracle.max_memory - used_memory:
                chosen_arms.append(selected_arm_id)
                used_memory += self.arms[selected_arm_id].memory
                if self.arms[selected_arm_id].table_name in table_count:
                    table_count[self.arms[selected_arm_id].table_name] += 1
                else:
                    table_count[self.arms[selected_arm_id].table_name] = 1
                # remove arms that are similar to max_ucb_arm_id
                # arm_class_dict = self.removed_covered_queries(arm_class_dict, selected_arm_id, bandit_arms)
                arm_dict = self.oracle.removed_covered_tables(arm_dict, selected_arm_id, self.arms, table_count)
                arm_dict = self.oracle.removed_covered_clusters(arm_dict, selected_arm_id, self.arms)
                arm_dict = self.oracle.removed_covered_queries_v2(arm_dict, selected_arm_id, self.arms)
                arm_dict = self.oracle.removed_covered_v2(arm_dict, selected_arm_id, self.arms,
                                                       self.oracle.max_memory - used_memory)
                arm_dict = self.oracle.removed_same_prefix(arm_dict, selected_arm_id, self.arms, 1)
            else:
                arm_dict.pop(selected_arm_id)
            max_count -= 1
            # Update contexts
            updated_context_dict = {}
            for arm_id, arm in arm_dict.items():
                updated_context_dict[arm_id] = context_dict[arm_id]
            context_dict = updated_context_dict
        return chosen_arms
    
    def get_arm_reward_hyp(self, arm):
        '''
        Calculate the hyp reward of arm in the current workload
        '''
        hyp_reward_dict = sql_helper.get_hyp_index_benefit_dict(arm, self.workload)
        arm.hyp_benefit = sum(hyp_reward_dict.values())
        return arm.hyp_benefit

    def label_input(self, input_vector):
        '''
        Given two candidate indexes, get the comparison label
        '''
        num_of_what_if_calls = 0
        # Get the hypothetical reward of each arm
        arm_1 = self.historical_arms[input_vector[0]]
        arm_2 = self.historical_arms[input_vector[1]]
        if arm_1.hyp_benefit == -1:
            self.get_arm_reward_hyp(arm_1)
            num_of_what_if_calls += 1
        if arm_2.hyp_benefit == -1:
            self.get_arm_reward_hyp(arm_2)
            num_of_what_if_calls += 1
        hyp_reward_1 = arm_1.hyp_benefit
        hyp_reward_2 = arm_2.hyp_benefit

        # Compare the rewards and generate label
        if hyp_reward_1 - hyp_reward_2 > 0.001:
            label = 1
        elif hyp_reward_1 - hyp_reward_2 < -0.001:
            label = -1
        else:
            label = 0
        
        return label, num_of_what_if_calls

    def update_training_data_hyp(self, round):
        rewards = []
        new_training_data = {}
        for i in range(len(self.arms)):
            hyp_reward = self.arms[i].hyp_benefit
            rewards.append(hyp_reward)
        for i in range(len(self.arms)):
            reward_1 = rewards[i]
            for j in range(len(self.arms)):
                reward_2 = rewards[j]
                if (reward_1 - reward_2 > self.alpha):
                    label = 1
                elif (reward_1 - reward_2 < -self.alpha):
                    label = -1
                else:
                    label = 0
                self.training_data[(self.arms[i].index_name, self.arms[j].index_name, round)] = label
                new_training_data[(self.arms[i].index_name, self.arms[j].index_name, round)] = label
        return new_training_data
    
    def initialize_vfdt(self, current_round):
        # self.training_data = {}
        # First, train clf with a small set of training data.(random select 10% of the candidate input data)
        current_candidate = [candidate for candidate in self.candidate_input_data if candidate[2] == current_round]
        size = int (0.2 * self.sample_rate * len(current_candidate))
        if size == 0:
                size = 1
        test_candidates = random.sample(current_candidate, int(self.sample_rate * len(current_candidate)))        
        label_time = 0
        training_time = 0
        if current_round == 0:
            training_data = {}
            selected_candidates = random.sample(current_candidate, size)
            start_label = time.time()
            num_of_what_if_calls = 0
            for candidate_input in selected_candidates:
                label, what_if_calls = self.label_input(candidate_input)
                training_data[candidate_input] = label
                num_of_what_if_calls += what_if_calls
            end_label = time.time()
            label_time += end_label - start_label
            start_train = time.time()
            self.retrain_clf(training_data)
            end_train = time.time()
            training_time += end_train - start_train
            # logging.info(f"Labeling time in initialize clf is {end_label - start_label}, num of what_if clalls {num_of_what_if_calls}")
            # logging.info(f"Training time in initialize clf is {end_train - end_label}, training size is {size}")
            # Delete selected candidates from self.candidate_input_data
            # self.candidate_input_data = [candidate for candidate in self.candidate_input_data if candidate not in selected_candidates]
            current_candidate = [candidate for candidate in current_candidate if candidate not in selected_candidates]

        for i in range(4):
            # Second, evaluate the uncertainty of clf on the rest training set, select the most uncertain ones.
            input_vectors = []
            for candidate_input in current_candidate:
                context_diff = self.context_dict[candidate_input[0]] - self.context_dict[candidate_input[1]]
                input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[candidate_input[2]]))
                input_vectors.append(input_vector)
            input_vectors_array = numpy.array(input_vectors)
            uncertain_indices = self.uncertainty_sampling(input_vectors_array, size)
            logging.info(f"Uncertain indices found,size is {len(uncertain_indices)}")

            # Third, label them.
            training_data = {}
            current_input_data = {}
            start_label = time.time()
            num_of_what_if_calls = 0
            for index in uncertain_indices:
                candidate_input = current_candidate[index]
                label, what_if_calls = self.label_input(candidate_input)
                num_of_what_if_calls += what_if_calls
                training_data[candidate_input] = label
                current_input_data[candidate_input] = label
            end_label = time.time()
            label_time += end_label - start_label

            # Fourth, retrain clf with the new training set & update the candidate set.
            start_train = time.time()
            self.retrain_clf(training_data)
            end_train = time.time()
            training_time += end_train - start_train
            current_candidate = [candidate for index, candidate in enumerate(current_candidate) if index not in uncertain_indices]

            # training_time += time.time() - end_label
        logging.info(f"Training time in uncertainty sampling is {training_time}")
        logging.info(f"Label time in uncertainty sampling is {label_time}")

        # Evaluate the accuracy of the ACLF
        test_inputs = []
        test_labels = []
        for test_candidate in test_candidates:
            context_diff = self.context_dict[test_candidate[0]] - self.context_dict[test_candidate[1]]
            input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[test_candidate[2]]))
            test_inputs.append(input_vector)
            if test_candidate in self.training_data:
                test_labels.append(self.training_data[test_candidate])
            else:
                label,_ = self.label_input(test_candidate)
                test_labels.append(label)
        for test_input, test_label in zip(test_inputs, test_labels):
            metric = metrics.Accuracy()
            input_vector_dict = {f'feature_{i}': value for i, value in enumerate(test_input)}
            y_pred = self.aclf.predict_one(input_vector_dict)
            if y_pred is None:
                y_pred = 0
            metric.update(y_pred, test_label)
        # accurracy = self.clf.score(test_inputs, test_labels)
        logging.info(f"Accuracy of the clf is {metric.get()}")



    def initialize_clf(self):
        '''
        initialize clf based on the hypothetical estimation of candidate indexes.
        Train clf with uncertainty sampling.
        '''
        # Clear the training data
        self.training_data = {}
        # First, train clf with a small set of training data.(random select 10% of the candidate input data)
        size = int (0.1 * len(self.candidate_input_data))
        training_time = 0
        label_time = 0
        if size == 0:
            size = 1
        selected_candidates = random.sample(self.candidate_input_data, size)
        test_candidates = random.sample(self.candidate_input_data, size)
        start_label = time.time()
        num_of_what_if_calls = 0
        for candidate_input in selected_candidates:
            label, what_if_calls = self.label_input(candidate_input)
            self.training_data[candidate_input] = label
            num_of_what_if_calls += what_if_calls
        end_label = time.time()
        self.retrain_clf(self.training_data)
        end_train = time.time()
        logging.info(f"Labeling time in initialize clf is {end_label - start_label}, num of what_if clalls {num_of_what_if_calls}")
        logging.info(f"Training time in initialize clf is {end_train - end_label}, training size is {size}")
        # Delete selected candidates from self.candidate_input_data
        self.candidate_input_data = [candidate for candidate in self.candidate_input_data if candidate not in selected_candidates]

        # Second, evaluate the uncertainty of clf on the rest training set, select the most uncertain ones.
        input_vectors = []
        for candidate_input in self.candidate_input_data:
            context_diff = self.context_dict[candidate_input[0]] - self.context_dict[candidate_input[1]]
            input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[candidate_input[2]]))
            input_vectors.append(input_vector)
        input_vectors_array = numpy.array(input_vectors)
        uncertain_indices = self.uncertainty_sampling(input_vectors_array, size)

        # Third, label them.
        current_input_data = {}
        start_label_time = time.time()
        num_of_what_if_calls = 0
        for index in uncertain_indices:
            candidate_input = self.candidate_input_data[index]
            label, what_if_calls = self.label_input(candidate_input)
            num_of_what_if_calls += what_if_calls
            self.training_data[candidate_input] = label
            current_input_data[candidate_input] = label
        end_label = time.time()
        label_time += end_label - start_label_time

        # Fourth, retrain clf with the new training set & update the candidate set.
        self.retrain_clf(self.training_data)
        self.candidate_input_data = [candidate for index, candidate in enumerate(self.candidate_input_data) if index not in uncertain_indices]

        training_time += time.time() - end_label
        logging.info(f"Training time in uncertainty sampling is {training_time}")
        logging.info(f"Label time in uncertainty sampling is {label_time}")
        
        # Finally, Evaluate the accuracy of the clf
        test_inputs = []
        test_labels = []
        for test_candidate in test_candidates:
            context_diff = self.context_dict[test_candidate[0]] - self.context_dict[test_candidate[1]]
            input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[test_candidate[2]]))
            test_inputs.append(input_vector)
            if test_candidate in self.training_data:
                test_labels.append(self.training_data[test_candidate])
            else:
                label,_ = self.label_input(test_candidate)
                test_labels.append(label)
        accurracy = self.clf.score(test_inputs, test_labels)
        logging.info(f"Accuracy of the clf is {accurracy}")


    def retrain_clf(self, new_training_data):
        input_vectors = []
        output_labels = []
        # for key, label in new_training_data.items():
        #     self.training_data[key] = label
        if self.classifier == "RandomForest":
            training_data = self.training_data
        elif self.classifier == "SGD":
            training_data = new_training_data
        elif self.classifier == "VFDT":
            training_data = new_training_data
        for key, label in training_data.items():
            context_diff = self.context_dict[key[0]] - self.context_dict[key[1]]
            input_vector = numpy.concatenate((context_diff, self.historical_workload_vectors[key[2]]))
            input_vectors.append(input_vector)
            output_labels.append(label)
        time_start = time.time()
        if self.classifier == "RandomForest":
            self.clf.fit(input_vectors, output_labels)
            # accuracy = self.clf.score(input_vectors, output_labels)
            # logging.info(f"Accuracy is {accuracy:.2f}")
        elif self.classifier == "SGD":
            scaler = StandardScaler()
            input_vectors_scaled = scaler.fit_transform(input_vectors)
            self.sgd_clf.fit(input_vectors_scaled, output_labels)
            accuracy = self.sgd_clf.score(input_vectors_scaled, output_labels)
            logging.info(f"Accuracy is {accuracy:.2f}")
        elif self.classifier == "VFDT":
            metric = metrics.Accuracy()
            input_vectors_dict = [
                {f'feature_{i}': value for i, value in enumerate(input_vector)}
                for input_vector in input_vectors
            ]
            for input_vector, output_label in zip(input_vectors_dict, output_labels):
                y_pred = self.aclf.predict_one(input_vector)
                if y_pred is None:
                    y_pred = 0
                self.aclf.learn_one(input_vector, output_label)
                metric.update(y_pred, output_label)
            logging.info(f'VFDT accuracy: {metric.get()}')
            # self.model = ensemble.AdaptiveRandomForestClassifier(
            #     model=tree.HoeffdingTreeClassifier(),
            #     n_models=10,  # Number of trees
            #     seed=42,
            #     )
            # train the model
            
        # self.clf.fit(input_vectors, output_labels)
        logging.info(f"Training time is {time.time() - time_start}")
    
    # def retrain_sgd_clf(self, training_data):
    #     self.sgd_clf.partial_fit(training_data[0], training_data[1], classes=[-1, 0, 1])




    def update(self, played_arms, reward, index_use):
        pass

    def update_v4(self, played_arms, arm_rewards):
        """
        This method can be used to update the reward after each play (improvements required)

        :param played_arms: list of played arms (super arm)
        :param arm_rewards: tuple (gains, creation cost) reward got form playing each arm
        """
        for i in played_arms:
            if self.arms[i].index_name in arm_rewards:
                arm_reward = arm_rewards[self.arms[i].index_name]
            else:
                arm_reward = (0, 0)
            logging.info(f"reward for {self.arms[i].index_name}, {self.arms[i].query_ids_backup} is {arm_reward}")
            self.arms[i].index_usage_last_batch = (self.arms[i].index_usage_last_batch + arm_reward[0]) / 2

            temp_context = numpy.zeros(self.context_vectors[i].shape)
            temp_context[1] = self.context_vectors[i][1]
            self.context_vectors[i][1] = 0

            self.v = self.v + (self.context_vectors[i] @ self.context_vectors[i].transpose())
            self.b = self.b + self.context_vectors[i] * arm_reward[0]

            self.v = self.v + (temp_context @ temp_context.transpose())
            self.b = self.b + temp_context * arm_reward[1]

        self.context_vectors = []
        self.upper_bounds = []

    def update_v5(self, played_arms, arm_rewards):
        """
        Update with workload-aware reward
        This method can be used to update the reward after each play (improvements required)

        :param played_arms: list of played arms (super arm)
        :param arm_rewards: tuple (gains, creation cost) reward got form playing each arm
        """
        for i in played_arms:
            if self.arms[i].index_name in arm_rewards:
                arm_reward = arm_rewards[self.arms[i].index_name]
            else:
                arm_reward = ({}, 0)
            logging.info(f"reward for {self.arms[i].index_name}, {self.arms[i].query_ids_backup} is {arm_reward}")
            self.arms[i].reward_query_ids = list(arm_reward[0].keys())
            self.arms[i].index_usage_last_batch = (self.arms[i].index_usage_last_batch + sum(arm_reward[0].values())) / 2

            temp_context = numpy.zeros(self.context_vectors[i].shape)
            temp_context[1] = self.context_vectors[i][1]
            self.context_vectors[i][1] = 0

            performance_reward = 0
            for reward in arm_reward[0].values():
                if reward < 0:
                    performance_reward += reward * constants.NEGATIVE_REWARD_FACTOR
                else:
                    performance_reward += reward

            self.v = self.v + (self.context_vectors[i] @ self.context_vectors[i].transpose())
            self.b = self.b + self.context_vectors[i] * performance_reward

            self.v = self.v + (temp_context @ temp_context.transpose())
            self.b = self.b + temp_context * arm_reward[1]

        self.context_vectors = []
        self.upper_bounds = []

    def update_v6(self, played_arms, arm_rewards, round):
        """
        Random Forest Update
        """
        # while not self.is_retrain[round]:
        #     round -= 1
        if round in self.cycle_workload_dict:
            round = self.cycle_workload_dict[round]
        # self.update_training_data_actual(played_arms, arm_rewards, round)
        # if classifier == "RandomForest":
        new_training_data = self.update_training_data_actual(played_arms, arm_rewards, round)
        self.retrain_clf(new_training_data)
        # elif classifier == "SGD":
        #     self.retrain_clf("SGD", training_data)
        # self.retrain_clf()
        # for i in played_arms:
        #     if self.arms[i].index_name in arm_rewards:
        #         arm_reward = arm_rewards[self.arms[i].index_name]
        #     else:
        #         arm_reward = ({}, 0)
        #     sum_value = sum(arm_reward[0].values())
        #     logging.info(f"Reward for {self.arms[i].index_name}, {self.arms[i].query_ids_backup} is {arm_reward}")
        # Old training set
        # context_vectors_array = numpy.array(context_vectors)
        # context_vectors = context_vectors_array.reshape(context_vectors_array.shape[0], -1)
        # workload_vector = numpy.array(workload_vector).reshape(1, -1)[0]
        
        # size = context_vectors.shape[0]
    def set_context_vectors(self, context_vectors, workload_vector):
        self.historical_workload_vectors.append(numpy.array(workload_vector).flatten())
        context_vectors_array = numpy.array(context_vectors)
        flattened_context_vectors = context_vectors_array.reshape(context_vectors_array.shape[0], -1)
        for i in range(len(self.arms)):
            if self.arms[i].index_name not in self.context_dict:
                self.context_dict[self.arms[i].index_name] = flattened_context_vectors[i] 
        #     if self.arms[i].index_name == "ix_catalog_sales_3":
        #         index_1 = i
        #     if self.arms[i].index_name == "ix_item_13_0_16":
        #         index_2 =i
        #     if index_1 and index_2:
        #         break
        # reward_in_training_set = training_dict[(index_1, index_2)]
        # logging.info(f"Reward for {self.arms[index_1].index_name} and {self.arms[index_2].index_name} in training set is {reward_in_training_set}")
        # Retraining
        # input_index_vectors = [context_vectors[context_pair[0]] - context_vectors[context_pair[1]] for context_pair in self.training_set.keys()]
        # input_vectors = [numpy.concatenate((input_index_vector, workload_vector)) for input_index_vector in input_index_vectors]
        # output_labels = list(self.training_set.values())
        # sample_weights = numpy.ones(len(output_labels))
        # sample_weights[indices_to_weight] = 5
        # start_time = time.time()
        # self.clf.fit(input_vectors, output_labels)
        # logging.info(f"Training time is {time.time() - start_time}")

        # input_vector = numpy.concatenate((context_vectors[index_1] - context_vectors[index_2], workload_vector))
        # reward_after_retraining = self.clf.predict([input_vector])[0]
        # logging.info(f"Reward for {self.arms[index_1].index_name} and {self.arms[index_2].index_name} after retraining is {reward_after_retraining} with input vector{input_vector}")
        

            
    def update_training_data_actual(self, played_arms, arm_rewards, round):
        length = len(played_arms)
        indices_to_weight = []
        new_training_data = {}
        for i in range(length):
            if self.arms[played_arms[i]].index_name in arm_rewards:
                arm_reward = arm_rewards[self.arms[played_arms[i]].index_name]
            else:
                arm_reward = ({}, 0)
            sum_value_1 = sum(arm_reward[0].values())
            logging.info(f"Reward for {self.arms[played_arms[i]].index_name}, {self.arms[played_arms[i]].query_ids_backup} is {arm_reward}")
            for j in range(len(self.arms)):
                if j in played_arms:
                    if self.arms[j].index_name in arm_rewards:
                        arm_reward = arm_rewards[self.arms[j].index_name]
                    else:
                        arm_reward = ({}, 0)
                    sum_value_2 = sum(arm_reward[0].values())
                    if (sum_value_1 - sum_value_2 > self.alpha):
                        label = 1
                    elif (sum_value_1 - sum_value_2 < -self.alpha):
                        label = -1
                    else:
                        label = 0
                    # old_label = self.training_data[(self.arms[played_arms[i]].index_name, self.arms[j].index_name, round)]
                    # old_label = self.predict_two_indexes(self.arms[played_arms[i]].index_name, self.arms[j].index_name)
                    # if label != old_label:
                    #     logging.info(f"The label for {self.arms[played_arms[i]].index_name} and {self.arms[j].index_name} is different, old label is {old_label}, new label is {label}")
                    self.training_data[(self.arms[played_arms[i]].index_name, self.arms[j].index_name, round)] = label
                    new_training_data[(self.arms[played_arms[i]].index_name, self.arms[j].index_name, round)] = label
                        # indices_to_weight.append(training_dict_index[(played_arms[i],j)])
                else:
                    if sum_value_1 > self.alpha:
                        label = 1
                    elif sum_value_1 < -self.alpha:
                        label = -1
                    else:
                        label = 0
                    # old_label = self.predict_two_indexes(self.arms[played_arms[i]].index_name, self.arms[j].index_name)
                    # if label != old_label:
                    #     logging.info(f"The label for {self.arms[played_arms[i]].index_name} and {self.arms[j].index_name} is different, old label is {old_label}, new label is {label}")
                    self.training_data[(self.arms[played_arms[i]].index_name, self.arms[j].index_name, round)] = label
                    self.training_data[(self.arms[j].index_name, self.arms[played_arms[i]].index_name, round)] = label * (-1)
                    new_training_data[(self.arms[played_arms[i]].index_name, self.arms[j].index_name, round)] = label
                    new_training_data[(self.arms[j].index_name, self.arms[played_arms[i]].index_name, round)] = label * (-1)
        return new_training_data

    def set_arms(self, bandit_arms):
        """
        This can be used to initially set the bandit arms in the algorithm

        :param bandit_arms: initial set of bandit arms
        :return:
        """
        self.arms = bandit_arms
        for arm in self.arms:
            if arm.index_name in self.historical_arms:
                arm.hyp_benefit_dict = self.historical_arms[arm.index_name].hyp_benefit_dict
            self.historical_arms[arm.index_name] = arm
    
    def set_candidate_input_data(self, current_round):
        """
        This method set the candidate input data for classifier training based on the bandit_arms.
        """
        for i in range(len(self.arms)):
            for j in range(len(self.arms)):
                self.candidate_input_data.append((self.arms[i].index_name, self.arms[j].index_name, current_round))
    
    def set_workload(self, query_obj_list_current):
        self.workload = query_obj_list_current

    def set_is_retrain(self, is_retrain):
        self.is_retrain = is_retrain

    def set_context_vectors(self, context_vectors, workload_vector):
        flattened_workload_vector = numpy.array(workload_vector).flatten()
        self.historical_workload_vectors.append(flattened_workload_vector)
        current_round = len(self.historical_workload_vectors) - 1
        for i in range(current_round):
            if numpy.array_equal(self.historical_workload_vectors[i], flattened_workload_vector):
                self.cycle_workload_dict[current_round] = i
                break
        context_vectors_array = numpy.array(context_vectors)
        flattened_context_vectors = context_vectors_array.reshape(context_vectors_array.shape[0], -1)
        for i in range(len(self.arms)):
            if self.arms[i].index_name not in self.context_dict:
                self.context_dict[self.arms[i].index_name] = flattened_context_vectors[i]


    def hard_reset(self):
        """
        Resets the bandit
        """
        self.hyper_alpha = self.alpha_original
        self.v = self.hyper_lambda * numpy.identity(self.context_size)  # identity matrix of n*n
        self.b = numpy.zeros((self.context_size, 1))  # [0, 0, ..., 0]T (column matrix) size = number of arms

    def workload_change_trigger(self, workload_change):
        """
        This forgets history based on the workload change

        :param workload_change: Percentage of new query templates added (0-1) 0: no workload change, 1: 100% shift
        """
        logging.info("Workload change identified " + str(workload_change))
        if workload_change > 0.5:
            self.hard_reset()
        else:
            forget_factor = 1 - workload_change * 2
            if workload_change > 0.1:
                self.hyper_alpha = self.alpha_original
            self.v = self.hyper_lambda * numpy.identity(self.context_size) + forget_factor * self.v
            self.b = forget_factor * self.b

