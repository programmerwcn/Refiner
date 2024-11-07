import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)
import query_embedding as query_embedding
from gensim.models import Word2Vec
import psycopg2
import json
import numpy as np
# import database.sql_connection as sql_connection
import database.sql_connection as sql_connection

def get_query_plan(query):
    conn = sql_connection.get_sql_connection()
    # conn = psycopg2.connect(
    #     dbname="tpch",
    #     user="wcn",
    #     password="12345ning",
    #     host="127.0.0.1",
    #     port="5434"
    # )
    cursor = conn.cursor()
    explain_query = f"EXPLAIN (FORMAT JSON) {query}"
    cursor.execute(explain_query)
    query_plan = cursor.fetchone()[0][0]["Plan"]
    cursor.close()
    conn.close()
    return query_plan

if __name__ == "__main__":
    work_file = "/home/wcn/indexAdvisor/ACCUCB-PostgreSQL/mab_selection/resources/workloads/tpc_h_static_100_pg.json"
    workload = []
    all_walks = []
    num_walks = 5
    walk_length = 10
    with open(work_file, "r") as rf:
            for line in rf.readlines():
                query = json.loads(line)
                workload.append(query["query_string"])
    for i in range(len(workload)):
        query = workload[i]
        plan = get_query_plan(query)
        root = query_embedding.parse_plan(plan)
        walks = query_embedding.generate_walks(root, num_walks, walk_length)
        all_walks.extend(walks)
    all_walks = [[str(node) for node in path] for path in all_walks]
    
    model = Word2Vec(all_walks, vector_size=10, window=3, min_count=1, sg=1, workers=4, epochs=10)

    node_embedddings = {node: model.wv[node] for node in model.wv.index_to_key}

    print("Node Embeddings:")
    for node, vector in  node_embedddings.items():
        print(f"{node}: {vector}")
    model.save("/home/wcn/indexAdvisor/ACCUCB-PostgreSQL/mab_selection/resources/tpch_node_embedding.model")
