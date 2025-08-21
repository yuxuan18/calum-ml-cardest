# different features used for plan nodes, filter columns etc. used for postgres plans

class PostgresTrueCardDetail:
    PLAN_FEATURES = ['act_card', 'est_width', 'workers_planned', 'op_name', 'act_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresEstSystemCardDetail:
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']


class PostgresDeepDBEstSystemCardDetail:
    PLAN_FEATURES = ['est_card', 'est_width', 'workers_planned', 'op_name', 'est_children_card']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['avg_width', 'correlation', 'data_type', 'n_distinct', 'null_frac']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples', 'relpages']

class KryptonEstSystemCardDetail:
    PLAN_FEATURES = ['est_card', 'op_name']
    FILTER_FEATURES = ['operator', 'literal_feature']
    COLUMN_FEATURES = ['data_type']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples']

class KryptonCardDetail:
    PLAN_FEATURES = ['op_name']
    FILTER_FEATURES = ['operator', 'r_literal']
    COLUMN_FEATURES = ['data_type']
    OUTPUT_COLUMN_FEATURES = ['aggregation']
    TABLE_FEATURES = ['reltuples']

class KryptonMultiCardDetail:
    JOIN_FEATURES = ['join_type']
    AGG_FEATURES = ['group_keys']
    SCAN_FEATURES = ['est_card']
    SET_FEATURES = ['set_op_type']
    LIMIT_FEATURES = ['n_limit']
    FILTER_FEATURES = ['operator', 'r_literal']
    COLUMN_FEATURES = ['data_type', 'column_id', 'tablename', 'table_size']
    # connections:
    # 1. Filter -> Join: join conditions or filter conditions
    # 2. Column -> Filter
    # 3. Join -> Agg
    # 4. Agg -> Join
    # 5. Scan -> Join
    # 6. Scan -> Agg
