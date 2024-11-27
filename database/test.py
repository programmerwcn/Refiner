import os
import re
import time
import copy
import json
import sys


import datetime
import logging
import configparser

import subprocess
import traceback
from collections import defaultdict


# 添加上级目录到 sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import constants
from database.query_plan import QueryPlan, QueryPlanPG
from database.column import Column
from database.table import Table
import shared.helper as helper

import database.sql_connection as sql_connection
import bandits.query_v5 as query_v5
import database.sql_helper_v2 as sql_helper

def test_18():
    conn = sql_connection.get_sql_connection()
    cur = conn.cursor()
    queries = helper.get_queries_v3()
    query_18 = queries[16]
    query = query_v5.Query(conn, 18, query_18['query_string'], query_18['predicates'],
                                  query_18['payload'], 0)
    sql_helper.create_query_drop_v4(conn, constants.SCHEMA_NAME, {}, {}, {}, [query])

test_18()