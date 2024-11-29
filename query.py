import database.sql_helper_v2 as sql_helper


class Query:
    def __init__(self, connection, query_id, query_string, predicates, payloads, time_stamp=0, freq=1):
        self.id = query_id
        self.predicates = predicates
        self.payload = payloads
        self.group_by = {}
        self.order_by = {}
        self.to_lower_case()
        # (0814): ?
        self.selectivity = sql_helper.get_selectivity_v3(connection, query_string, self.predicates)
        self.query_string = query_string
        # (1016): newly added.
        self.freq = freq
        self.frequency = 1
        self.last_seen = time_stamp
        self.first_seen = time_stamp
        self.table_scan_times = sql_helper.get_table_scan_times_structure(connection)
        self.index_scan_times = sql_helper.get_table_scan_times_structure(connection)
        self.table_scan_times_hyp = sql_helper.get_table_scan_times_structure(connection)
        self.index_scan_times_hyp = sql_helper.get_table_scan_times_structure(connection)
        self.context = None
        self.exe_time = -1

    # def __init__(self, connection, query_id, query_string, predicates, payloads, groupby, orderby, time_stamp=0, freq=1):
    #     self.id = query_id
    #     self.predicates = predicates
    #     self.payload = payloads
    #     self.group_by = groupby
    #     self.order_by = orderby
    #     self.to_lower_case()
    #     # (0814): ?
    #     self.selectivity = sql_helper.get_selectivity_v3(connection, query_string, self.predicates)
    #     self.query_string = query_string
    #     # (1016): newly added.
    #     self.freq = freq
    #     self.frequency = 1
    #     self.last_seen = time_stamp
    #     self.first_seen = time_stamp
    #     self.table_scan_times = sql_helper.get_table_scan_times_structure(connection)
    #     self.index_scan_times = sql_helper.get_table_scan_times_structure(connection)
    #     self.table_scan_times_hyp = sql_helper.get_table_scan_times_structure(connection)
    #     self.index_scan_times_hyp = sql_helper.get_table_scan_times_structure(connection)
    #     self.context = None
        

    def __hash__(self):
        return self.id
    
    def set_goupby_orderby(self,groupby,orderby):
        self.group_by = groupby
        self.order_by = orderby
        self.to_lower_case()


    def get_id(self):
        return self.id
    
    def to_lower_case(self):
        lower_case_predicates = {}
        lower_case_payloads = {}
        lower_case_groupbys = {}
        lower_case_orderbys = {}
        for table, payloads in self.payload.items():
            lower_case_payloads[table.lower()] = [payload.lower() for payload in payloads]
        for table, predicates in self.predicates.items():       
            lower_case_predicates[table.lower()] = [predicate.lower() for predicate in predicates]
        for table, groupbys in self.group_by.items():
            lower_case_groupbys[table.lower()] = [groupby.lower() for groupby in groupbys]
        for table, orderbys in self.order_by.items():
            lower_case_orderbys[table.lower()] = [orderby.lower() for orderby in orderbys]
        self.predicates = lower_case_predicates
        self.payload = lower_case_payloads
        self.group_by = lower_case_groupbys
        self.order_by = lower_case_orderbys

