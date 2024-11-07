from gensim.models import Word2Vec
import psycopg2
import numpy as np
import json
import Levenshtein 
import embedding_utils.query_embedding as query_embedding
import database.sql_connection as sql_connection
def get_query_plan(query):
    # conn = psycopg2.connect(
    #     dbname="tpcds",
    #     user="wtz",
    #     password="wtz*888",
    #     host="localhost",
    #     port="5436"
    # )
    conn = sql_connection.get_sql_connection()
    cursor = conn.cursor()
    explain_query = f"EXPLAIN (FORMAT JSON) {query}"
    cursor.execute(explain_query)
    query_plan = cursor.fetchone()[0][0]["Plan"]
    cursor.close()
    conn.close()
    return query_plan

def infer(query, embedding_model_path):
    model = Word2Vec.load(embedding_model_path)
    plan = get_query_plan(query)
    root = query_embedding.parse_plan(plan)
    walk = query_embedding.generate_walks(root, 1, 10)[0]
    walk = [str(node) for node in walk]
    node_embedding_list = []
    for node in walk:
        if node in model.wv:
            node_embedding = model.wv[node]
        else:
            most_similar_node = find_most_similar_node(node, model.wv.index_to_key)
            node_embedding = model.wv[most_similar_node]
        node_embedding_list.append(node_embedding)
    # 将路径中每个节点嵌入的均值作为整个路径的嵌入
    path_embedding = np.mean(node_embedding_list, axis=0)
    return path_embedding

def infer_v2(query_id, embedding_path):
    embedding_path = "/home/wcn/indexAdvisor/ACCUCB-PostgreSQL/mab_selection/resources/tpch_query_embeddings.json"
    embeddings = {}
    #load json
    with open(embedding_path, "r") as rf:
        embeddings = json.load(rf)
    return embeddings[query_id]



def find_most_similar_node(node, nodes):
    most_similar_node = None
    node_parts = node.split("_", maxsplit=1)
    node_type = node_parts[0]
    node_info = node_parts[1]
    similarity = dict() 
    for n in nodes:
        n_parts = n.split("_", maxsplit=1)
        n_type = n_parts[0]
        n_info = n_parts[1]
        if n_type != node_type:
            similarity[n] = 0
        else:
            similarity[n] = Levenshtein.ratio(node_info, n_info)
    most_similar_node = max(similarity, key=similarity.get)
    return most_similar_node


# if __name__ == "__main__":
#     work_file = "/home/csf/baseline/workload/tpc_h_static_100_pg_new.json"
#     workload = []
#     with open(work_file, "r") as rf:
#             for line in rf.readlines():
#                 query = json.loads(line)
#                 workload.append(query["query_string"])
#     for i in range(10, 20):
#         query = workload[i]
#         path_embedding = infer(query)
#         print(f"{query}: {path_embedding}")
