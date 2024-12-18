import operator
from abc import abstractmethod
import numpy


import constants


class BaseOracle:

    def __init__(self, max_memory):
        self.max_memory = max_memory

    @abstractmethod
    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms):
        pass

    @staticmethod
    def arm_satisfy_predicate(arm, predicate, table_name):
        """
        Check if the bandit arm can be helpful for a given predicate
        :param arm: bandit arm
        :param predicate: predicate that we wanna test against
        :param table_name: table name
        :return: boolean
        """
        if table_name == arm.table_name and predicate == arm.index_cols[0]:
            return True
        return False

    @staticmethod
    def removed_covered_v2(arm_ucb_dict, chosen_id, bandit_arms, remaining_memory):
        """
        second version which is based on the remaining memory and already chosen arms

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param remaining_memory: max_memory - used_memory
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id] <= bandit_arms[chosen_id] or bandit_arms[arm_id].memory > remaining_memory):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_tables(arm_ucb_dict, chosen_id, bandit_arms, table_count):
        """

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param table_count: count of indexes already chosen for each table
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and table_count[
                    bandit_arms[arm_id].table_name] >= constants.MAX_INDEXES_PER_TABLE):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_clusters(arm_ucb_dict, chosen_id, bandit_arms):
        """

        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if not (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and bandit_arms[
                chosen_id].cluster is not None and bandit_arms[arm_id].cluster is not None and bandit_arms[
                        arm_id].cluster == bandit_arms[chosen_id].cluster):
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_queries_v2(arm_ucb_dict, chosen_id, bandit_arms):
        """
        When covering index is selected for a query we gonna remove all other arms from that query
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            query_ids = bandit_arms[chosen_id].query_ids
            for query_id in query_ids:
                if (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and bandit_arms[
                        chosen_id].is_include == 1 and query_id in bandit_arms[arm_id].query_ids):
                    bandit_arms[arm_id].query_ids.remove(query_id)
            if bandit_arms[arm_id].query_ids != set():
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_covered_queries(arm_ucb_dict, chosen_id, bandit_arms):
        """
        Only allow one effective index for each query
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :return: reduced arm list
        """
        arms_to_remove = {}
        for arm_id in arm_ucb_dict:
            for query_id in arm_ucb_dict[chosen_id].reward_query_ids:
                if (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and
                        query_id in bandit_arms[arm_id].reward_query_ids):
                    arms_to_remove[arm_id] = arm_ucb_dict[arm_id]
        reduced_arm_ucb_dict = {k: v for k, v in arm_ucb_dict.items() if k not in arms_to_remove}
        return reduced_arm_ucb_dict
                    
                    


    @staticmethod
    def removed_low_expected_rewards(arm_ucb_dict, threshold):
        """
        It make sense to remove arms with low expected reward
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param threshold: expected reward threshold
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        for arm_id in arm_ucb_dict:
            if arm_ucb_dict[arm_id] > threshold:
                reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict

    @staticmethod
    def removed_same_prefix(arm_ucb_dict, chosen_id, bandit_arms, prefix_length):
        """
        One index for one query for table
        :param arm_ucb_dict: dictionary of arms and upper confidence bounds
        :param chosen_id: chosen arm in this round
        :param bandit_arms: Bandit arm list
        :param prefix_length: This is the length of the prefix
        :return: reduced arm list
        """
        reduced_arm_ucb_dict = {}
        if len(bandit_arms[chosen_id].index_cols) < prefix_length:
            return arm_ucb_dict
        else:
            for arm_id in arm_ucb_dict:
                if (bandit_arms[arm_id].table_name == bandit_arms[chosen_id].table_name and
                        len(bandit_arms[arm_id].index_cols) > prefix_length):
                    for i in range(prefix_length):
                        if bandit_arms[arm_id].index_cols[i] != bandit_arms[chosen_id].index_cols[i]:
                            reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
                            continue
                else:
                    reduced_arm_ucb_dict[arm_id] = arm_ucb_dict[arm_id]
        return reduced_arm_ucb_dict


class OracleV7(BaseOracle):

    '''
    This method greedily selects the best arm based on the upper confidence bound.
    It also tries to ensure that the selected arms are not covering each other.
    '''
    def get_super_arm(self, upper_bounds, context_vectors, bandit_arms, budget):
        used_memory = 0
        chosen_arms = []
        arm_ucb_dict = {}
        table_count = {}

        for i in range(len(bandit_arms)):
            arm_ucb_dict[i] = upper_bounds[i]

        arm_ucb_dict = self.removed_low_expected_rewards(arm_ucb_dict, 0)

        max_count = budget
        # current_count = 0

        while len(arm_ucb_dict) > 0 and max_count > 0:
            # greedily select the arm with the maximum upper confidence bound
            max_ucb_arm_id = max(arm_ucb_dict.items(), key=operator.itemgetter(1))[0]
            if bandit_arms[max_ucb_arm_id].memory < self.max_memory - used_memory:
                chosen_arms.append(max_ucb_arm_id)
                used_memory += bandit_arms[max_ucb_arm_id].memory
                if bandit_arms[max_ucb_arm_id].table_name in table_count:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] += 1
                else:
                    table_count[bandit_arms[max_ucb_arm_id].table_name] = 1
                # remove arms that are similar to max_ucb_arm_id
                arm_ucb_dict = self.removed_covered_tables(arm_ucb_dict, max_ucb_arm_id, bandit_arms, table_count)
                arm_ucb_dict = self.removed_covered_clusters(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                arm_ucb_dict = self.removed_covered_queries_v2(arm_ucb_dict, max_ucb_arm_id, bandit_arms)
                arm_ucb_dict = self.removed_covered_v2(arm_ucb_dict, max_ucb_arm_id, bandit_arms,
                                                       self.max_memory - used_memory)
                arm_ucb_dict = self.removed_same_prefix(arm_ucb_dict, max_ucb_arm_id, bandit_arms, 1)
            else:
                arm_ucb_dict.pop(max_ucb_arm_id)
            max_count -= 1

        return chosen_arms

    def epsilon_greedy(self, predicated_classes, context_vectors, bandit_arms):
        used_memory = 0
        chosen_arms = []
        arm_class_dict = {}
        table_count = {}

        for i in range(len(bandit_arms)):
            arm_class_dict[i] = predicated_classes[i] + 1

        # arm_ucb_dict = self.removed_low_expected_rewards(arm_ucb_dict, 0)

        max_count = 5
        # max_class = max(predicated_classes)
        # current_count = 0
        while len(arm_class_dict) > 0 and max_count > 0:
            current_prob = [prob / sum(arm_class_dict.values()) for prob in arm_class_dict.values()]
            selected_arm_id = numpy.random.choice(list(arm_class_dict.keys()), p=current_prob)
            if bandit_arms[selected_arm_id].memory < self.max_memory - used_memory:
                chosen_arms.append(selected_arm_id)
                used_memory += bandit_arms[selected_arm_id].memory
                if bandit_arms[selected_arm_id].table_name in table_count:
                    table_count[bandit_arms[selected_arm_id].table_name] += 1
                else:
                    table_count[bandit_arms[selected_arm_id].table_name] = 1
                # remove arms that are similar to max_ucb_arm_id
                # arm_class_dict = self.removed_covered_queries(arm_class_dict, selected_arm_id, bandit_arms)
                arm_class_dict = self.removed_covered_tables(arm_class_dict, selected_arm_id, bandit_arms, table_count)
                arm_class_dict = self.removed_covered_clusters(arm_class_dict, selected_arm_id, bandit_arms)
                arm_class_dict = self.removed_covered_queries_v2(arm_class_dict, selected_arm_id, bandit_arms)
                arm_class_dict = self.removed_covered_v2(arm_class_dict, selected_arm_id, bandit_arms,
                                                       self.max_memory - used_memory)
                arm_class_dict = self.removed_same_prefix(arm_class_dict, selected_arm_id, bandit_arms, 1)
            else:
                arm_class_dict.pop(selected_arm_id)
            max_count -= 1

        return chosen_arms