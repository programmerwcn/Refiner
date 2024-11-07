import os
import logging

# ===============================  Program Related  ===============================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_FOLDER = "experiments"
EXPERIMENT_CONFIG = "/config/exp.conf"
DB_CONFIG = "/config/db.conf"
WORKLOADS_FOLDER = "/resources/workloads"
LOGGING_LEVEL = logging.INFO

TABLE_SCAN_TIME_LENGTH = 1000

# ===============================  Database / Workload  ===============================
SCHEMA_NAME = "dbo"
HYP_RATIO = 1000000
TIME_OUT = 100000

# ===============================  Arm Generation Heuristics  ===============================
INDEX_INCLUDES = 1
MAX_PERMUTATION_LENGTH = 3
SMALL_TABLE_IGNORE = 10000
TABLE_MIN_SELECTIVITY = 0.2
PREDICATE_MIN_SELECTIVITY = 0.01
# (0814): newly added.
MAX_INDEX_WIDTH = 3

# ===============================  Bandit Parameters  ===============================
ALPHA_REDUCTION_RATE = 1.05
QUERY_MEMORY = 1
BANDIT_FORGET_FACTOR = 0.6
MAX_INDEXES_PER_TABLE = 6
CREATION_COST_REDUCTION_FACTOR = 5
STOP_EXPLORATION_ROUND = 500
UNIFORM_ASSUMPTION_START = 10

# ===============================  Reward Related  ===============================
COST_TYPE_ELAPSED_TIME = 1
COST_TYPE_CPU_TIME = 2
COST_TYPE_SUB_TREE_COST = 3
COST_TYPE_CURRENT_EXECUTION = COST_TYPE_ELAPSED_TIME
COST_TYPE_CURRENT_CREATION = COST_TYPE_ELAPSED_TIME
NEGATIVE_REWARD_FACTOR = 10

# COST_TYPE_CURRENT_EXECUTION = COST_TYPE_SUB_TREE_COST
# COST_TYPE_CURRENT_CREATION = COST_TYPE_SUB_TREE_COST

# ===============================  Context Related  ===============================
CONTEXT_UNIQUENESS = 0
CONTEXT_INCLUDES = False
STATIC_CONTEXT_SIZE = 3

# ===============================  Reporting Related  ===============================
DF_COL_COMP_ID = "Component"
DF_COL_REP = "Rep"
DF_COL_BATCH = "Batch Number"
DF_COL_BATCH_COUNT = "# of Batches"
DF_COL_MEASURE_NAME = "Measurement Name"
DF_COL_MEASURE_VALUE = "Measurement Value"

MEASURE_TOTAL_WORKLOAD_TIME = "Total Workload Time"
MEASURE_INDEX_CREATION_COST = "Index Creation Time"
MEASURE_INDEX_RECOMMENDATION_COST = "Index Recommendation Cost"
MEASURE_QUERY_EXECUTION_COST = "Query Execution Cost"
MEASURE_MEMORY_COST = "Memory Cost"
MEASURE_BATCH_TIME = "Batch Time"
MEASURE_HYP_BATCH_TIME = "Hyp Batch Time"

COMPONENT_MAB = "MAB"
COMPONENT_TA_OPTIMAL = "TA_OPTIMAL"
COMPONENT_TA_FULL = "TA_FULL"
COMPONENT_TA_CURRENT = "TA_CURRENT"
COMPONENT_TA_SCHEDULE = "TA_SCHEDULE"
COMPONENT_OPTIMAL = "OPTIMAL"
COMPONENT_NO_INDEX = "NO_INDEX"
COMPONENT_DDQN = "DDQN"
COMPONENT_DDQN_SINGLE_COLUMN = "DDQN_SINGLE_COLUMN"
COMPONENT_ACC_UCB = "ACC_UCB"

TA_WORKLOAD_TYPE_OPTIMAL = "optimal"
TA_WORKLOAD_TYPE_FULL = "full"
TA_WORKLOAD_TYPE_CURRENT = "current"
TA_WORKLOAD_TYPE_SCHEDULE = "schedule"

# ===============================  Other  ===============================
TABLE_SCAN_TIMES = {"SSB": {"customer": [], "dwdate": [], "lineorder": [], "part": [], "supplier": []},
                    "TPCH": {"lineitem": [], "customer": [], "nation": [], "orders": [], "part": [], "partsupp": [],
                             "region": [], "supplier": []},
                    "TPCHSKEW": {"LINEITEM": [], "CUSTOMER": [], "NATION": [], "ORDERS": [], "PART": [], "PARTSUPP": [],
                                 "REGION": [], "SUPPLIER": []},
                    "TPCDS": {"call_center": [], "catalog_page": [], "catalog_returns": [], "catalog_sales": [],
                              "customer": [], "customer_address": [], "customer_demographics": [], "date_dim": [],
                              "dbgen_version": [], "household_demographics": [], "income_band": [], "inventory": [],
                              "item": [], "promotion": [], "reason": [], "ship_mode": [], "store": [],
                              "store_returns": [], "store_sales": [], "time_dim": [], "warehouse": [], "web_page": [],
                              "web_returns": [], "web_sales": [], "web_site": []},
                    "IMDB": {"aka_name": [], "aka_title": [], "cast_info": [], "char_name": [],
                             "comp_cast_type": [], "company_name": [], "company_type": [], "complete_cast": [],
                             "info_type": [], "keyword": [], "kind_type": [], "link_type": [],
                             "movie_companies": [], "movie_info": [], "movie_info_idx": [], "movie_keyword": [],
                             "movie_link": [],
                             "name": [], "person_info": [], "role_type": [], "title": []}}
TABLE_ALIAS = {"TPCDS": {'dbgen_version': {'dv_create_date': 0, 'dv_create_time': 1, 'dv_version': 2, 'dv_cmdline_args': 3}, 'customer_address': {'ca_gmt_offset': 0, 'ca_address_sk': 1, 'ca_street_number': 2, 'ca_street_name': 3, 'ca_street_type': 4, 'ca_suite_number': 5, 'ca_city': 6, 'ca_county': 7, 'ca_state': 8, 'ca_zip': 9, 'ca_country': 10, 'ca_location_type': 11, 'ca_address_id': 12}, 'customer_demographics': {'cd_dep_college_count': 0, 'cd_purchase_estimate': 1, 'cd_dep_count': 2, 'cd_dep_employed_count': 3, 'cd_demo_sk': 4, 'cd_gender': 5, 'cd_marital_status': 6, 'cd_education_status': 7, 'cd_credit_rating': 8}, 'date_dim': {'d_quarter_seq': 0, 'd_year': 1, 'd_first_dom': 2, 'd_last_dom': 3, 'd_same_day_ly': 4, 'd_same_day_lq': 5, 'd_dow': 6, 'd_moy': 7, 'd_dom': 8, 'd_qoy': 9, 'd_date_sk': 10, 'd_fy_year': 11, 'd_fy_quarter_seq': 12, 'd_fy_week_seq': 13, 'd_date': 14, 'd_month_seq': 15, 'd_week_seq': 16, 'd_current_year': 17, 'd_date_id': 18, 'd_day_name': 19, 'd_quarter_name': 20, 'd_holiday': 21, 'd_weekend': 22, 'd_following_holiday': 23, 'd_current_day': 24, 'd_current_week': 25, 'd_current_month': 26, 'd_current_quarter': 27}, 'warehouse': {'w_warehouse_sk': 0, 'w_warehouse_sq_ft': 1, 'w_gmt_offset': 2, 'w_country': 3, 'w_street_number': 4, 'w_street_name': 5, 'w_street_type': 6, 'w_suite_number': 7, 'w_city': 8, 'w_county': 9, 'w_state': 10, 'w_zip': 11, 'w_warehouse_id': 12, 'w_warehouse_name': 13}, 'ship_mode': {'sm_ship_mode_sk': 0, 'sm_ship_mode_id': 1, 'sm_type': 2, 'sm_code': 3, 'sm_carrier': 4, 'sm_contract': 5}, 'time_dim': {'t_time_sk': 0, 't_time': 1, 't_hour': 2, 't_minute': 3, 't_second': 4, 't_meal_time': 5, 't_time_id': 6, 't_am_pm': 7, 't_shift': 8, 't_sub_shift': 9}, 'reason': {'r_reason_sk': 0, 'r_reason_id': 1, 'r_reason_desc': 2}, 'income_band': {'ib_income_band_sk': 0, 'ib_lower_bound': 1, 'ib_upper_bound': 2}, 'item': {'i_item_sk': 0, 'i_rec_start_date': 1, 'i_rec_end_date': 2, 'i_current_price': 3, 'i_wholesale_cost': 4, 'i_brand_id': 5, 'i_class_id': 6, 'i_category_id': 7, 'i_manufact_id': 8, 'i_manager_id': 9, 'i_class': 10, 'i_container': 11, 'i_category': 12, 'i_item_id': 13, 'i_product_name': 14, 'i_manufact': 15, 'i_item_desc': 16, 'i_size': 17, 'i_formulation': 18, 'i_color': 19, 'i_brand': 20, 'i_units': 21}, 'store': {'s_tax_precentage': 0, 's_rec_start_date': 1, 's_rec_end_date': 2, 's_closed_date_sk': 3, 's_number_employees': 4, 's_floor_space': 5, 's_market_id': 6, 's_division_id': 7, 's_company_id': 8, 's_gmt_offset': 9, 's_store_sk': 10, 's_geography_class': 11, 's_market_desc': 12, 's_market_manager': 13, 's_zip': 14, 's_division_name': 15, 's_country': 16, 's_company_name': 17, 's_street_number': 18, 's_store_id': 19, 's_street_name': 20, 's_street_type': 21, 's_suite_number': 22, 's_store_name': 23, 's_city': 24, 's_county': 25, 's_hours': 26, 's_manager': 27, 's_state': 28}, 'call_center': {'cc_call_center_sk': 0, 'cc_rec_start_date': 1, 'cc_rec_end_date': 2, 'cc_closed_date_sk': 3, 'cc_open_date_sk': 4, 'cc_employees': 5, 'cc_sq_ft': 6, 'cc_mkt_id': 7, 'cc_division': 8, 'cc_company': 9, 'cc_gmt_offset': 10, 'cc_tax_percentage': 11, 'cc_state': 12, 'cc_mkt_class': 13, 'cc_mkt_desc': 14, 'cc_market_manager': 15, 'cc_zip': 16, 'cc_division_name': 17, 'cc_country': 18, 'cc_company_name': 19, 'cc_call_center_id': 20, 'cc_street_number': 21, 'cc_street_name': 22, 'cc_street_type': 23, 'cc_suite_number': 24, 'cc_name': 25, 'cc_class': 26, 'cc_city': 27, 'cc_county': 28, 'cc_hours': 29, 'cc_manager': 30}, 'customer': {'c_customer_sk': 0, 'c_current_hdemo_sk': 1, 'c_current_addr_sk': 2, 'c_first_shipto_date_sk': 3, 'c_first_sales_date_sk': 4, 'c_birth_day': 5, 'c_birth_month': 6, 'c_birth_year': 7, 'c_last_review_date_sk': 8, 'c_current_cdemo_sk': 9, 'c_customer_id': 10, 'c_last_name': 11, 'c_preferred_cust_flag': 12, 'c_birth_country': 13, 'c_login': 14, 'c_email_address': 15, 'c_salutation': 16, 'c_first_name': 17}, 'web_site': {'web_tax_percentage': 0, 'web_rec_start_date': 1, 'web_rec_end_date': 2, 'web_open_date_sk': 3, 'web_close_date_sk': 4, 'web_mkt_id': 5, 'web_company_id': 6, 'web_gmt_offset': 7, 'web_site_sk': 8, 'web_zip': 9, 'web_mkt_class': 10, 'web_mkt_desc': 11, 'web_market_manager': 12, 'web_country': 13, 'web_company_name': 14, 'web_street_number': 15, 'web_street_name': 16, 'web_street_type': 17, 'web_site_id': 18, 'web_suite_number': 19, 'web_city': 20, 'web_name': 21, 'web_county': 22, 'web_state': 23, 'web_class': 24, 'web_manager': 25}, 'store_returns': {'sr_returned_date_sk': 0, 'sr_return_time_sk': 1, 'sr_item_sk': 2, 'sr_customer_sk': 3, 'sr_cdemo_sk': 4, 'sr_hdemo_sk': 5, 'sr_addr_sk': 6, 'sr_store_sk': 7, 'sr_reason_sk': 8, 'sr_ticket_number': 9, 'sr_return_quantity': 10, 'sr_return_amt': 11, 'sr_return_tax': 12, 'sr_return_amt_inc_tax': 13, 'sr_fee': 14, 'sr_return_ship_cost': 15, 'sr_refunded_cash': 16, 'sr_reversed_charge': 17, 'sr_store_credit': 18, 'sr_net_loss': 19}, 'household_demographics': {'hd_demo_sk': 0, 'hd_income_band_sk': 1, 'hd_dep_count': 2, 'hd_vehicle_count': 3, 'hd_buy_potential': 4}, 'web_page': {'wp_max_ad_count': 0, 'wp_char_count': 1, 'wp_link_count': 2, 'wp_image_count': 3, 'wp_web_page_sk': 4, 'wp_rec_start_date': 5, 'wp_rec_end_date': 6, 'wp_creation_date_sk': 7, 'wp_access_date_sk': 8, 'wp_customer_sk': 9, 'wp_web_page_id': 10, 'wp_url': 11, 'wp_autogen_flag': 12, 'wp_type': 13}, 'promotion': {'p_end_date_sk': 0, 'p_item_sk': 1, 'p_cost': 2, 'p_response_target': 3, 'p_start_date_sk': 4, 'p_promo_sk': 5, 'p_channel_tv': 6, 'p_channel_radio': 7, 'p_channel_press': 8, 'p_channel_event': 9, 'p_channel_demo': 10, 'p_channel_details': 11, 'p_purpose': 12, 'p_discount_active': 13, 'p_promo_id': 14, 'p_promo_name': 15, 'p_channel_dmail': 16, 'p_channel_email': 17, 'p_channel_catalog': 18}, 'catalog_page': {'cp_catalog_number': 0, 'cp_catalog_page_number': 1, 'cp_end_date_sk': 2, 'cp_catalog_page_sk': 3, 'cp_start_date_sk': 4, 'cp_type': 5, 'cp_catalog_page_id': 6, 'cp_department': 7, 'cp_description': 8}, 'inventory': {'inv_date_sk': 0, 'inv_item_sk': 1, 'inv_warehouse_sk': 2, 'inv_quantity_on_hand': 3}, 'catalog_returns': {'cr_returned_date_sk': 0, 'cr_returned_time_sk': 1, 'cr_item_sk': 2, 'cr_refunded_customer_sk': 3, 'cr_refunded_cdemo_sk': 4, 'cr_refunded_hdemo_sk': 5, 'cr_refunded_addr_sk': 6, 'cr_returning_customer_sk': 7, 'cr_returning_cdemo_sk': 8, 'cr_returning_hdemo_sk': 9, 'cr_returning_addr_sk': 10, 'cr_call_center_sk': 11, 'cr_catalog_page_sk': 12, 'cr_ship_mode_sk': 13, 'cr_warehouse_sk': 14, 'cr_reason_sk': 15, 'cr_order_number': 16, 'cr_return_quantity': 17, 'cr_return_amount': 18, 'cr_return_tax': 19, 'cr_return_amt_inc_tax': 20, 'cr_fee': 21, 'cr_return_ship_cost': 22, 'cr_refunded_cash': 23, 'cr_reversed_charge': 24, 'cr_store_credit': 25, 'cr_net_loss': 26}, 'web_returns': {'wr_returned_date_sk': 0, 'wr_returned_time_sk': 1, 'wr_item_sk': 2, 'wr_refunded_customer_sk': 3, 'wr_refunded_cdemo_sk': 4, 'wr_refunded_hdemo_sk': 5, 'wr_refunded_addr_sk': 6, 'wr_returning_customer_sk': 7, 'wr_returning_cdemo_sk': 8, 'wr_returning_hdemo_sk': 9, 'wr_returning_addr_sk': 10, 'wr_web_page_sk': 11, 'wr_reason_sk': 12, 'wr_order_number': 13, 'wr_return_quantity': 14, 'wr_return_amt': 15, 'wr_return_tax': 16, 'wr_return_amt_inc_tax': 17, 'wr_fee': 18, 'wr_return_ship_cost': 19, 'wr_refunded_cash': 20, 'wr_reversed_charge': 21, 'wr_account_credit': 22, 'wr_net_loss': 23}, 'web_sales': {'ws_sold_date_sk': 0, 'ws_sold_time_sk': 1, 'ws_ship_date_sk': 2, 'ws_item_sk': 3, 'ws_bill_customer_sk': 4, 'ws_bill_cdemo_sk': 5, 'ws_bill_hdemo_sk': 6, 'ws_bill_addr_sk': 7, 'ws_ship_customer_sk': 8, 'ws_ship_cdemo_sk': 9, 'ws_ship_hdemo_sk': 10, 'ws_ship_addr_sk': 11, 'ws_web_page_sk': 12, 'ws_web_site_sk': 13, 'ws_ship_mode_sk': 14, 'ws_warehouse_sk': 15, 'ws_promo_sk': 16, 'ws_order_number': 17, 'ws_quantity': 18, 'ws_wholesale_cost': 19, 'ws_list_price': 20, 'ws_sales_price': 21, 'ws_ext_discount_amt': 22, 'ws_ext_sales_price': 23, 'ws_ext_wholesale_cost': 24, 'ws_ext_list_price': 25, 'ws_ext_tax': 26, 'ws_coupon_amt': 27, 'ws_ext_ship_cost': 28, 'ws_net_paid': 29, 'ws_net_paid_inc_tax': 30, 'ws_net_paid_inc_ship': 31, 'ws_net_paid_inc_ship_tax': 32, 'ws_net_profit': 33}, 'catalog_sales': {'cs_sold_date_sk': 0, 'cs_sold_time_sk': 1, 'cs_ship_date_sk': 2, 'cs_bill_customer_sk': 3, 'cs_bill_cdemo_sk': 4, 'cs_bill_hdemo_sk': 5, 'cs_bill_addr_sk': 6, 'cs_ship_customer_sk': 7, 'cs_ship_cdemo_sk': 8, 'cs_ship_hdemo_sk': 9, 'cs_ship_addr_sk': 10, 'cs_call_center_sk': 11, 'cs_catalog_page_sk': 12, 'cs_ship_mode_sk': 13, 'cs_warehouse_sk': 14, 'cs_item_sk': 15, 'cs_promo_sk': 16, 'cs_order_number': 17, 'cs_quantity': 18, 'cs_wholesale_cost': 19, 'cs_list_price': 20, 'cs_sales_price': 21, 'cs_ext_discount_amt': 22, 'cs_ext_sales_price': 23, 'cs_ext_wholesale_cost': 24, 'cs_ext_list_price': 25, 'cs_ext_tax': 26, 'cs_coupon_amt': 27, 'cs_ext_ship_cost': 28, 'cs_net_paid': 29, 'cs_net_paid_inc_tax': 30, 'cs_net_paid_inc_ship': 31, 'cs_net_paid_inc_ship_tax': 32, 'cs_net_profit': 33}, 'store_sales': {'ss_sold_date_sk': 0, 'ss_sold_time_sk': 1, 'ss_item_sk': 2, 'ss_customer_sk': 3, 'ss_cdemo_sk': 4, 'ss_hdemo_sk': 5, 'ss_addr_sk': 6, 'ss_store_sk': 7, 'ss_promo_sk': 8, 'ss_ticket_number': 9, 'ss_quantity': 10, 'ss_wholesale_cost': 11, 'ss_list_price': 12, 'ss_sales_price': 13, 'ss_ext_discount_amt': 14, 'ss_ext_sales_price': 15, 'ss_ext_wholesale_cost': 16, 'ss_ext_list_price': 17, 'ss_ext_tax': 18, 'ss_coupon_amt': 19, 'ss_net_paid': 20, 'ss_net_paid_inc_tax': 21, 'ss_net_profit': 22}},
               "TPCH": {
    "nation": {
        "n_nationkey": 0,
        "n_regionkey": 1,
        "n_name": 2,
        "n_comment": 3
    },
    "region": {
        "r_regionkey": 4,
        "r_name": 5,
        "r_comment": 6
    },
    "part": {
        "p_size": 7,
        "p_retailprice": 8,
        "p_partkey": 9,
        "p_brand": 10,
        "p_type": 11,
        "p_container": 12,
        "p_comment": 13,
        "p_name": 14,
        "p_mfgr": 15
    },
    "supplier": {
        "s_nationkey": 16,
        "s_suppkey": 17,
        "s_acctbal": 18,
        "s_comment": 19,
        "s_phone": 20,
        "s_name": 21,
        "s_address": 22
    },
    "partsupp": {
        "ps_partkey": 23,
        "ps_suppkey": 24,
        "ps_availqty": 25,
        "ps_supplycost": 26,
        "ps_comment": 27
    },
    "customer": {
        "c_acctbal": 28,
        "c_nationkey": 29,
        "c_custkey": 30,
        "c_phone": 31,
        "c_mktsegment": 32,
        "c_comment": 33,
        "c_name": 34,
        "c_address": 35
    },
    "orders": {
        "o_orderkey": 36,
        "o_custkey": 37,
        "o_totalprice": 38,
        "o_orderdate": 39,
        "o_shippriority": 40,
        "o_orderpriority": 41,
        "o_orderstatus": 42,
        "o_clerk": 43,
        "o_comment": 44
    },
    "lineitem": {
        "l_commitdate": 45,
        "l_receiptdate": 46,
        "l_linenumber": 47,
        "l_quantity": 48,
        "l_orderkey": 49,
        "l_extendedprice": 50,
        "l_discount": 51,
        "l_tax": 52,
        "l_partkey": 53,
        "l_suppkey": 54,
        "l_shipdate": 55,
        "l_comment": 56,
        "l_returnflag": 57,
        "l_linestatus": 58,
        "l_shipinstruct": 59,
        "l_shipmode": 60
    },
    "hypopg_list_indexes": {
        "indexrelid": 61,
        "index_name": 62,
        "schema_name": 63,
        "table_name": 64,
        "am_name": 65
    }
},
"IMDB":
{
    "aka_title": {
        "episode_of_id": 5,
        "season_nr": 6,
        "episode_nr": 7,
        "kind_id": 8,
        "id": 9,
        "production_year": 10,
        "movie_id": 11,
        "md5sum": 12,
        "title": 13,
        "imdb_index": 14,
        "phonetic_code": 15,
        "note": 16
    },
    "movie_info_idx": {
        "id": 17,
        "movie_id": 18,
        "info_type_id": 19,
        "info": 20,
        "note": 21
    },
    "role_type": {
        "id": 22,
        "role": 23
    },
    "title": {
        "episode_of_id": 24,
        "season_nr": 25,
        "episode_nr": 26,
        "production_year": 27,
        "id": 28,
        "imdb_id": 29,
        "kind_id": 30,
        "md5sum": 31,
        "title": 32,
        "imdb_index": 33,
        "phonetic_code": 34,
        "series_years": 35
    },
    "aka_name": {
        "id": 36,
        "person_id": 37,
        "name": 38,
        "imdb_index": 39,
        "name_pcode_cf": 40,
        "name_pcode_nf": 41,
        "surname_pcode": 42,
        "md5sum": 43
    },
    "comp_cast_type": {
        "id": 44,
        "kind": 45
    },
    "company_name": {
        "imdb_id": 46,
        "id": 47,
        "country_code": 48,
        "md5sum": 49,
        "name_pcode_nf": 50,
        "name_pcode_sf": 51,
        "name": 52
    },
    "company_type": {
        "id": 53,
        "kind": 54
    },
    "complete_cast": {
        "id": 55,
        "movie_id": 56,
        "subject_id": 57,
        "status_id": 58
    },
    "info_type": {
        "id": 59,
        "info": 60
    },
    "keyword": {
        "id": 61,
        "keyword": 62,
        "phonetic_code": 63
    },
    "kind_type": {
        "id": 64,
        "kind": 65
    },
    "link_type": {
        "id": 66,
        "link": 67
    },
    "movie_companies": {
        "id": 68,
        "movie_id": 69,
        "company_id": 70,
        "company_type_id": 71,
        "note": 72
    },
    "movie_keyword": {
        "id": 73,
        "movie_id": 74,
        "keyword_id": 75
    },
    "movie_link": {
        "id": 76,
        "movie_id": 77,
        "linked_movie_id": 78,
        "link_type_id": 79
    },
    "movie_info": {
        "id": 80,
        "movie_id": 81,
        "info_type_id": 82,
        "info": 83,
        "note": 84
    },
    "person_info": {
        "id": 85,
        "person_id": 86,
        "info_type_id": 87,
        "info": 88,
        "note": 89
    },
    "name": {
        "imdb_id": 90,
        "id": 91,
        "imdb_index": 92,
        "gender": 93,
        "name_pcode_cf": 94,
        "name_pcode_nf": 95,
        "surname_pcode": 96,
        "md5sum": 97,
        "name": 98
    },
    "char_name": {
        "imdb_id": 99,
        "id": 100,
        "imdb_index": 101,
        "md5sum": 102,
        "name_pcode_nf": 103,
        "surname_pcode": 104,
        "name": 105
    },
    "cast_info": {
        "person_role_id": 106,
        "person_id": 107,
        "movie_id": 108,
        "id": 109,
        "role_id": 110,
        "nr_order": 111,
        "note": 112
    }
}
}