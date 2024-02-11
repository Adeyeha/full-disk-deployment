import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv() 

db_path = os.getenv('db_name')
table_names = [os.getenv('all_records'), os.getenv('most_recent_record')]

def create_table_sql(table_name):
    return f'''
    CREATE TABLE {table_name} (
        source_date TEXT,
        obs_date TEXT,
        raw_filename TEXT,
        noaa_ar_filename TEXT,
        local_request_date TEXT,
        error TEXT,
        flare_probability FLOAT,
        non_flare_probability FLOAT,
        explanation TEXT
    )
    '''.strip()

def backup_and_recreate_table(cursor, table_name):
    today = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_table_name = f"{table_name}_bkp_{today}"
    cursor.execute(f"ALTER TABLE {table_name} RENAME TO {backup_table_name};")
    print(f"Backed up '{table_name}' to '{backup_table_name}'.")
    cursor.execute(create_table_sql(table_name))
    print(f"Table '{table_name}' has been recreated with the correct schema.")

def validate_schema(cursor, table_name):
    cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
    row = cursor.fetchone()
    if row:
        actual_create_table_sql = row[0].strip()
        expected_sql = create_table_sql(table_name)
        return actual_create_table_sql == expected_sql
    return False

def create_or_validate_db(table_name):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        table_exists = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';").fetchone() is not None
        
        if table_exists:
            if not validate_schema(cursor, table_name):
                print(f"Schema for '{table_name}' does not match. Backing up and recreating.")
                backup_and_recreate_table(cursor, table_name)
            else:
                print(f"Table '{table_name}' exists and matches the expected schema.")
        else:
            print(f"Table '{table_name}' does not exist. Creating.")
            cursor.execute(create_table_sql(table_name))
            print(f"Table '{table_name}' has been created.")

if __name__ == '__main__':
    for table_name in table_names:
        if table_name:  # Checking that the environment variable is not None
            create_or_validate_db(table_name)
