import constants
import configparser
import copy
db_config = configparser.ConfigParser()
db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
db_type = db_config["SYSTEM"]["db_type"]
database = db_config[db_type]["database"]
# table_scan_times_hyp = copy.deepcopy(constants.TABLE_SCAN_TIMES[database[:-4]])
table_alias = copy.deepcopy(constants.TABLE_ALIAS[database.upper()])

class BanditArm:
    def __init__(self, index_cols, table_name, memory, table_row_count, include_cols=()):
        self.schema_name = 'dbo'
        self.table_name = table_name
        self.index_cols = index_cols
        self.include_cols = include_cols
        index_cols_alas = []
        for index_col in index_cols:
           index_cols_alas.append(str(table_alias[table_name][index_col]))
        if self.include_cols:
            include_cols_alas = []
            for include_col in include_cols:
                include_cols_alas.append(str(table_alias[table_name][include_col]))
            # include_col_hash = hashlib.sha1('_'.join(include_cols).lower().encode()).hexdigest()
            include_col_names = '_'.join(tuple(map(lambda x: x[0:4], include_cols_alas)))
            self.index_name = 'ixn_' + table_name + '_' + '_'.join(index_cols_alas) + 'include_' + include_col_names
        else:
            self.index_name = 'ix_' + table_name + '_' + '_'.join(index_cols_alas)
        # self.index_name = self.index_name[:127]
        self.memory = memory
        self.table_row_count = table_row_count
        self.name_encoded_context = []
        self.index_usage_last_batch = 0

        # (0814): LINEITEM_1_all? table_name + '_' + str(query_id) + '_all'?
        self.cluster = None
        self.query_id = None
        self.query_ids = set()
        self.query_ids_backup = set()
        self.reward_query_ids = list()
        self.is_include = 0
        self.arm_value = {}
        self.clustered_index_time = 0
        self.potentials = {}

        # (0813): newly added.
        self.oid = None
        self.hyp_benefit = -1
        self.hyp_benefit_dict = {}
        self.upper_bound = 0

    def __eq__(self, other):
        return self.index_name == other.index_name

    def __hash__(self):
        return hash(self.index_name)

    def __le__(self, other):
        if len(self.index_cols) > len(other.index_cols):
            return False
        else:
            for i in range(len(self.index_cols)):
                if self.index_cols[i] != other.index_cols[i]:
                    return False
            # (0814): consumed by other (prefix)
            return True

    def __str__(self):
        return self.index_name

    @staticmethod
    def get_arm_id(index_cols, table_name, include_cols=()):
        if include_cols:
            include_col_names = '_'.join(tuple(map(lambda x: x[0:4], include_cols))).lower()
            arm_id = 'ixn_' + table_name + '_' + '_'.join(index_cols) + '_' + include_col_names
        else:
            arm_id = 'ix_' + table_name + '_' + '_'.join(index_cols)
        arm_id = arm_id[:127]
        return arm_id
