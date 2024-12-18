import itertools

import numpy

import constants as constants
import database.sql_helper_v2 as sql_helper
from bandit_arm import BanditArm
import embedding_utils.infer_embedding as infer_embedding

# (0815): the universal set of all the arms generated
bandit_arm_store = dict()


def gen_arms_from_predicates_v2(connection, query_obj):
    """
    This method take predicates (a dictionary of lists) as input and
    creates the generate arms for all possible column combinations

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """
    bandit_arms = dict()
    predicates = query_obj.predicates
    payloads = query_obj.payload
    query_id = query_obj.id
    tables = sql_helper.get_tables(connection)

    # arms based on the predicates, todo(0814): permutations
    # 对每个table涉及到的predicates，生成permutation
    for table_name, table_predicates in predicates.items():
        # (0801): newly added.
        # print(tables.keys())
        table = tables[table_name.lower()]
        includes = list()
        if table_name in payloads:
            includes = list(set(payloads[table_name]) - set(table_predicates))
        # if table.table_row_count < constants.SMALL_TABLE_IGNORE or (
        #         query_obj.selectivity[table_name] > constants.TABLE_MIN_SELECTIVITY and len(includes) > 0):
        # (0814): newly modified.
        if table.table_row_count < constants.SMALL_TABLE_IGNORE:
            continue

        col_permutations = list()
        # if len(table_predicates) > constants.MAX_INDEX_WIDTH:
        #     table_predicates = table_predicates[:constants.MAX_INDEX_WIDTH]
        # (0814): newly added.
        index_width = len(table_predicates)
        if index_width > constants.MAX_INDEX_WIDTH:
            index_width = constants.MAX_INDEX_WIDTH
        for j in range(1, (index_width + 1)):
            col_permutations = col_permutations + list(itertools.permutations(table_predicates, j))
        for col_permutation in col_permutations:
            arm_id = BanditArm.get_arm_id(col_permutation, table_name)
            table_row_count = table.table_row_count
            # (0815): for what?
            arm_value = (1 - query_obj.selectivity[table_name]) * (
                    len(col_permutation) / len(table_predicates)) * table_row_count
            if arm_id in bandit_arm_store:
                bandit_arm = bandit_arm_store[arm_id]
                bandit_arm.potentials[query_id] = query_obj.exe_time
                bandit_arm.query_id = query_id
                if query_id in bandit_arm.arm_value:
                    bandit_arm.arm_value[query_id] += arm_value
                    bandit_arm.arm_value[query_id] /= 2
                else:
                    bandit_arm.arm_value[query_id] = arm_value
            else:
                size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                 table_name, col_permutation)
                if size == 0: 
                    continue
                # index_cols, table_name, memory, table_row_count, include_cols=()
                bandit_arm = BanditArm(col_permutation, table_name, size, table_row_count)
                bandit_arm.potentials[query_id] = query_obj.exe_time
                bandit_arm.query_id = query_id
                if len(col_permutation) == len(table_predicates):
                    bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                    if len(includes) == 0:
                        bandit_arm.is_include = 1
                bandit_arm.arm_value[query_id] = arm_value
                bandit_arm_store[arm_id] = bandit_arm

            # (1016): newly modified.
            if bandit_arm.index_name not in bandit_arms:
                bandit_arms[arm_id] = bandit_arm

    # arms based on the payloads
    for table_name, table_payloads in payloads.items():
        # (1016): newly added.
        if table_name.lower() not in predicates:
            table = tables[table_name.lower()]
            if table.table_row_count < constants.SMALL_TABLE_IGNORE:
                continue

            # (0815): newly added.
            if len(table_payloads) > constants.MAX_INDEX_WIDTH:
                continue

            col_permutation = table_payloads
            arm_id = BanditArm.get_arm_id(col_permutation, table_name)
            table_row_count = table.table_row_count
            arm_value = 0.001 * table_row_count
            if arm_id in bandit_arm_store:
                bandit_arm = bandit_arm_store[arm_id]
                bandit_arm.potentials[query_id] = query_obj.exe_time
                bandit_arm.query_id = query_id
                if query_id in bandit_arm.arm_value:
                    bandit_arm.arm_value[query_id] += arm_value
                    bandit_arm.arm_value[query_id] /= 2
                else:
                    bandit_arm.arm_value[query_id] = arm_value
            else:
                size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                 table_name, col_permutation)
                if size == 0: 
                    continue
                bandit_arm = BanditArm(col_permutation, table_name, size, table_row_count)
                bandit_arm.potentials[query_id] = query_obj.exe_time
                bandit_arm.query_id = query_id
                bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                bandit_arm.is_include = 1
                bandit_arm.arm_value[query_id] = arm_value
                bandit_arm_store[arm_id] = bandit_arm
            # (1016): newly modified.
            if bandit_arm.index_name not in bandit_arms:
                bandit_arms[arm_id] = bandit_arm

    if constants.INDEX_INCLUDES:
        for table_name, table_predicates in predicates.items():
            # (0801): newly added.
            table = tables[table_name.lower()]
            if table.table_row_count < constants.SMALL_TABLE_IGNORE:
                continue
            includes = list()
            if table_name in payloads:
                includes = sorted(list(set(payloads[table_name]) - set(table_predicates)))
            if includes:
                col_permutations = list(itertools.permutations(table_predicates, len(table_predicates)))
                for col_permutation in col_permutations:
                    arm_id_with_include = BanditArm.get_arm_id(col_permutation, table_name, includes)
                    table_row_count = table.table_row_count
                    arm_value = (1 - query_obj.selectivity[table_name]) * table_row_count
                    if arm_id_with_include not in bandit_arm_store:
                        size_with_includes = sql_helper.get_estimated_size_of_index_v1(connection,
                                                                                       constants.SCHEMA_NAME,
                                                                                       table_name,
                                                                                       col_permutation +
                                                                                       tuple(includes))
                        if size_with_includes == 0: 
                           continue
                        bandit_arm = BanditArm(col_permutation, table_name, size_with_includes,
                                               table_row_count, includes)
                        bandit_arm.potentials[query_id] = query_obj.exe_time
                        bandit_arm.is_include = 1
                        bandit_arm.query_id = query_id
                        bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                        bandit_arm.arm_value[query_id] = arm_value
                        bandit_arm_store[arm_id_with_include] = bandit_arm
                    else:
                        bandit_arm_store[arm_id_with_include].query_id = query_id
                        bandit_arm_store[arm_id_with_include].potentials[query_id] = query_obj.exe_time
                        if query_id in bandit_arm_store[arm_id_with_include].arm_value:
                            bandit_arm_store[arm_id_with_include].arm_value[query_id] += arm_value
                            bandit_arm_store[arm_id_with_include].arm_value[query_id] /= 2
                        else:
                            bandit_arm_store[arm_id_with_include].arm_value[query_id] = arm_value
                    bandit_arms[arm_id_with_include] = bandit_arm_store[arm_id_with_include]

    return bandit_arms

def gen_arms_from_predicates_v3(connection, query_obj):
    """
    Generate Candidate index based on permutations of predicates & groupBy & OrderBy

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """
    bandit_arms = gen_arms_from_predicates_v2(connection, query_obj)
    tables = sql_helper.get_tables(connection)

    for table_name, groupbys in query_obj.group_by.items():
        table = tables[table_name]
        # Generate new arm
        arm_id = BanditArm.get_arm_id(groupbys, table_name)
        table_row_count = table.table_row_count
        arm_value = (1 - query_obj.selectivity[table_name]) * table_row_count
        if arm_id in bandit_arms: # If the arm already exists
            bandit_arm = bandit_arms[arm_id]
            bandit_arm.query_id = query_obj.id
            if query_obj.id in bandit_arm.arm_value:
                bandit_arm.arm_value[query_obj.id] += arm_value
                bandit_arm.arm_value[query_obj.id] /= 2
            else:
                bandit_arm.arm_value[query_obj.id] = arm_value
        else:
            size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                table_name, groupbys)
            if size == 0: 
                continue
            # index_cols, table_name, memory, table_row_count, include_cols=()
            bandit_arm = BanditArm(groupbys, table_name, size, table_row_count)
            bandit_arm.query_id = query_obj.id
            
            bandit_arm.cluster = table_name + '_' + str(query_obj.id) + '_all'
            bandit_arm.arm_value[query_obj.id] = arm_value
            bandit_arms[arm_id] = bandit_arm

        
    return bandit_arms


def gen_arms_from_predicates_single(connection, query_obj):
    """
    This method take predicates (a dictionary of lists) as input and creates the generate arms for all possible
    column combinations

    :param connection: SQL connection
    :param query_obj: Query object
    :return: list of bandit arms
    """
    bandit_arms = dict()
    predicates = query_obj.predicates
    query_id = query_obj.id
    tables = sql_helper.get_tables(connection)
    includes = list()
    for table_name, table_predicates in predicates.items():
        table = tables[table_name]
        if table.table_row_count < 1000 or (
                query_obj.selectivity[table_name] > constants.TABLE_MIN_SELECTIVITY and len(includes) > 0):
            continue
        col_permutations = list()
        if len(table_predicates) > 6:
            table_predicates = table_predicates[0:6]
        col_permutations = col_permutations + list(itertools.permutations(table_predicates, 1))
        for col_permutation in col_permutations:
            arm_id = BanditArm.get_arm_id(col_permutation, table_name)
            table_row_count = table.table_row_count
            arm_value = (1 - query_obj.selectivity[table_name]) * (
                    len(col_permutation) / len(table_predicates)) * table_row_count
            if arm_id in bandit_arm_store:
                bandit_arm = bandit_arm_store[arm_id]
                bandit_arm.query_id = query_id
                if query_id in bandit_arm.arm_value:
                    bandit_arm.arm_value[query_id] += arm_value
                    bandit_arm.arm_value[query_id] /= 2
                else:
                    bandit_arm.arm_value[query_id] = arm_value
            else:
                size = sql_helper.get_estimated_size_of_index_v1(connection, constants.SCHEMA_NAME,
                                                                 table_name, col_permutation)
                bandit_arm = BanditArm(col_permutation, table_name, size, table_row_count)
                bandit_arm.query_id = query_id
                if len(col_permutation) == len(table_predicates):
                    bandit_arm.cluster = table_name + '_' + str(query_id) + '_all'
                    if len(includes) == 0:
                        bandit_arm.is_include = 1
                bandit_arm.arm_value[query_id] = arm_value
                bandit_arm_store[arm_id] = bandit_arm
            if bandit_arm not in bandit_arms:
                bandit_arms[arm_id] = bandit_arm

    return bandit_arms


# ========================== Context Vectors ==========================


def get_predicate_position(arm, predicate, table_name):
    """
    Returns float between 0 and 1  if a arm includes a predicate for the the correct table

    :param arm: bandit arm
    :param predicate: given predicate
    :param table_name: table name
    :return: float [0, 1]
    """
    for i in range(len(arm.index_cols)):
        # (0814): newly modified.
        if table_name == arm.table_name and predicate == arm.index_cols[i]:
            return i
        # if table_name.upper() == arm.table_name and predicate.upper() == arm.index_cols[i]:
        #     return i
    return -1


def get_context_vector_v2(bandit_arm, all_columns, context_size, uniqueness=0, includes=False):
    """
    Return the context vector for a given arm, and set of predicates. Size of the context vector will depend on
    the arm and the set of predicates (for now on predicates)

    :param bandit_arm: bandit arm
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: a context vector
    """
    context_vectors = dict()
    for j in range(uniqueness):
        context_vectors[j] = numpy.zeros((context_size, 1), dtype=float)
    left_over_context = numpy.zeros((context_size, 1), dtype=float)
    include_context = numpy.zeros((context_size, 1), dtype=float)

    if len(bandit_arm.name_encoded_context) > 0:
        context_vector = bandit_arm.name_encoded_context
    else:
        i = 0
        for table_name in all_columns:
            for k in range(len(all_columns[table_name])):
                column_position_in_arm = get_predicate_position(bandit_arm, all_columns[table_name][k], table_name)
                if column_position_in_arm >= 0:
                    if column_position_in_arm < uniqueness:
                        context_vectors[column_position_in_arm][i] = 1
                    else:
                        left_over_context[i] = 1 / (10 ** column_position_in_arm)
                elif all_columns[table_name][k] in bandit_arm.include_cols:
                    include_context[i] = 1
                i += 1

        full_list = list()
        for j in range(uniqueness):
            full_list = full_list + list(context_vectors[j])
        full_list = full_list + list(left_over_context)
        if includes:
            full_list = full_list + list(include_context)
        context_vector = numpy.array(full_list, ndmin=2, dtype=float)
        bandit_arm.name_encoded_context = context_vector

    return context_vector


def get_name_encode_context_vectors_v2(bandit_arm_dict, all_columns, context_size, uniqueness=0, includes=False):
    """
    Return the context vectors for a given arms, and set of predicates.

    :param bandit_arm_dict: bandit arms
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: list of context vectors
    """
    context_vectors = list()
    for key, bandit_arm in bandit_arm_dict.items():
        context_vector = get_context_vector_v2(bandit_arm, all_columns, context_size, uniqueness, includes)
        context_vectors.append(context_vector)

    return context_vectors


def get_derived_value_context_vectors_v3(connection, bandit_arm_dict, query_obj_list,
                                         chosen_arms_last_round, with_includes):
    """
    Similar to the v2, but it don't have the is_include part

    :param connection: SQL connection
    :param bandit_arm_dict: bandit arms
    :param query_obj_list: list of queries
    :param chosen_arms_last_round: Already created arms
    :param with_includes: have is include feature, note if includes are added to encode part we don't need it here.
    :return: list of context vectors
    """
    context_vectors = list()
    database_size = sql_helper.get_database_size(connection)
    for key, bandit_arm in bandit_arm_dict.items():
        keys_last_round = set(chosen_arms_last_round.keys())
        if bandit_arm.index_name not in keys_last_round:
            index_size = bandit_arm.memory
        else:
            index_size = 0
        context_vector = numpy.array([
            bandit_arm.index_usage_last_batch,
            index_size / database_size,
            bandit_arm.is_include if with_includes else 0,
        ], ndmin=2).transpose()
        context_vectors.append(context_vector)
    return context_vectors

def get_derived_value_context_vectors_v4(connection, bandit_arm_dict, query_obj_list,
                                         chosen_arms_last_round, with_includes):
    """
    在statistical中加入了arm_value的考量
    arm_value = (1 - query_obj.selectivity[table_name]) * table_row_count

    :param connection: SQL connection
    :param bandit_arm_dict: bandit arms
    :param query_obj_list: list of queries
    :param chosen_arms_last_round: Already created arms
    :param with_includes: have is include feature, note if includes are added to encode part we don't need it here.
    :return: list of context vectors
    """
    context_vectors = list()
    database_size = sql_helper.get_database_size(connection)
    for key, bandit_arm in bandit_arm_dict.items():
        keys_last_round = set(chosen_arms_last_round.keys())
        if bandit_arm.index_name not in keys_last_round:
            index_size = bandit_arm.memory
        else:
            index_size = 0
        max_arm_value_id = max(bandit_arm.arm_value, key=bandit_arm.arm_value.get)
        max_arm_value = bandit_arm.arm_value[max_arm_value_id]
        context_vector = numpy.array([
            bandit_arm.index_usage_last_batch,
            index_size / database_size,
            bandit_arm.is_include if with_includes else 0,
            max_arm_value], ndmin=2).transpose()
        context_vectors.append(context_vector)
    return context_vectors

def get_derived_value_context_vectors_v5(connection, bandit_arm_dict, query_obj_list,
                                         chosen_arms_last_round, with_includes):
    """
    在statistical中加入了arm_value & query cost的考量
    arm_value = (1 - query_obj.selectivity[table_name]) * table_row_count

    :param connection: SQL connection
    :param bandit_arm_dict: bandit arms
    :param query_obj_list: list of queries
    :param chosen_arms_last_round: Already created arms
    :param with_includes: have is include feature, note if includes are added to encode part we don't need it here.
    :return: list of context vectors
    """
    context_vectors = list()
    database_size = sql_helper.get_database_size(connection)
    for key, bandit_arm in bandit_arm_dict.items():
        keys_last_round = set(chosen_arms_last_round.keys())
        if bandit_arm.index_name not in keys_last_round:
            index_size = bandit_arm.memory
        else:
            index_size = 0
        max_arm_value_id = max(bandit_arm.arm_value, key=bandit_arm.arm_value.get)
        max_arm_value = bandit_arm.arm_value[max_arm_value_id]
        potential = sum(bandit_arm.potentials.values())
        context_vector = numpy.array([
            bandit_arm.index_usage_last_batch,
            index_size / database_size,
            bandit_arm.is_include if with_includes else 0,
            max_arm_value,
            potential], ndmin=2).transpose()
        context_vectors.append(context_vector)
    return context_vectors

def get_workload_embedding_context_vectors(bandit_arm_dict, workload, workload_size, embedding_model_path):
    """
    Return the context vectors for a given arms, and set of predicates.

    :param bandit_arm_dict: bandit arms
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: list of context vectors
    """

    context_vectors = list()
    query_embeddings = list()
    count = 0
    for query in workload:
        # query_embedding = infer(query.query_string,embedding_model_path)
        query_embedding = infer_embedding.infer_v2(query.id,"")
        query_embeddings.append(query_embedding)
        count += 1
        if count >= workload_size:
            break
    # concatenate the embeddings
    # workload_context = numpy.concatenate(query_embeddings, axis=0)
    
    workload_context = numpy.mean(query_embeddings, axis=0).reshape(-1, 1)
    # for key, bandit_arm in bandit_arm_dict.items():
    #     context_vectors.append(workload_context)
    
    # context_vectors = list()
    # workload_context = numpy.zeros((workload_size, 1), dtype=float)
    # for query in workload:
    #     workload_context[int(query.id)-1] = 1

    # for key, bandit_arm in bandit_arm_dict.items():
    #     context_vectors.append(workload_context)

    return workload_context

def comp_workload_distance(past_workload_context, current_workload_context):
    # Calculate the jaccard similarity between two workload context vectors
    # intersection = numpy.sum((past_workload_context & current_workload_context)!=0)
    # union = numpy.sum((past_workload_context | current_workload_context)!=0)
    # return intersection / union
    # return numpy.linalg.norm(past_workload_context - current_workload_context)
    return numpy.dot(past_workload_context.T, current_workload_context) / (numpy.linalg.norm(past_workload_context) * numpy.linalg.norm(current_workload_context))

def get_workload_encode_context_vectors(bandit_arm_dict, workload, workload_size):
    """
    Return the context vectors for a given arms, and set of predicates.

    :param bandit_arm_dict: bandit arms
    :param all_columns: predicate dict(list)
    :param context_size: size of the context vector
    :param uniqueness: how many columns in the index to consider when considering the context
    :param includes: add includes to the arm encode
    :return: list of context vectors
    """
    context_vectors = list()
    workload_context = numpy.zeros((workload_size, 1), dtype=float)
    for query in workload:
        workload_context[int(query.id)-1] = 1

    # for key, bandit_arm in bandit_arm_dict.items():
    #     context_vectors.append(workload_context)

    return workload_context

def get_query_context_v1(query_object, all_columns, context_size):
    """
    Return the context vectors for a given query. (column vector, corresponding column in query -> 1)

    :param query_object: query object
    :param all_columns: columns in database
    :param context_size: size of the context
    :return: list of context vectors
    """
    context_vector = numpy.zeros((context_size, 1), dtype=float)
    if query_object.context is not None:
        context_vector = query_object.context
    else:
        i = 0
        for table_name in all_columns:
            for k in range(len(all_columns[table_name])):
                # (0814): newly modified.
                context_vector[i] = 1 if table_name.upper() in query_object.predicates and \
                                         all_columns[table_name][k].upper() in \
                                         query_object.predicates[table_name.upper()] else 0
                i += 1
        query_object.context = context_vector
    return context_vector
