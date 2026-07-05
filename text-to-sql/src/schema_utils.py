import json


def build_schemas(tables_path: str) -> dict:
    """
    Args:
        tables_path: path to tables.json

    Returns:
        dict mapping db_id -> schema string
    """
    tables = json.load(open(tables_path))

    db_schemas = {}

    for db in tables:
        db_id     = db["db_id"]
        tbl_names = db["table_names_original"]
        col_names = db["column_names_original"]
        col_types = db["column_types"]

        table_cols = {}

        for i, (tbl_idx, col_name) in enumerate(col_names):
            if tbl_idx == -1:
                continue  # skip the special "*" column

            tbl      = tbl_names[tbl_idx]
            col_type = col_types[i]

            table_cols.setdefault(tbl, []).append(f"{col_name}:{col_type}")

        parts = [
            f"{tbl} ( {' , '.join(cols)} )"
            for tbl, cols in table_cols.items()
        ]

        db_schemas[db_id] = " | ".join(parts)

    return db_schemas
