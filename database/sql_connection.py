import psycopg2
import pyodbc
import configparser

import sys
import os

# 获取当前文件的目录路径
current_directory = os.path.dirname(__file__)

# 获取上级目录的路径
parent_directory = os.path.abspath(os.path.join(current_directory, os.pardir))

# 将上级目录添加到 sys.path
if parent_directory not in sys.path:
    sys.path.append(parent_directory)

import constants


def drop_connection(connection):
    # connection = get_sql_connection()
    cursor = connection.cursor()
    try:
        # Retrieve all indexes starting with 'IX'
        cursor.execute("""
               SELECT indexname as index_name, tablename as table_name
               FROM pg_indexes 
               WHERE schemaname='public'    
        """)
        indexes = cursor.fetchall()
        connection.autocommit = False
        # Drop each index found
        for index_name, table_name in indexes:
            if "_pkey" not in index_name and "primary" not in index_name:
                drop_sql = f"DROP INDEX IF EXISTS {index_name}"
                cursor.execute(drop_sql)
                print(f"Dropped index: {index_name} on {table_name}")
        # Commit the changes
        connection.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        # Roll back in case of error
        connection.rollback()

    finally:
        connection.autocommit = True
        # Clean up
        # cursor.close()
        # connection.close()
        # print("Completed dropping indexes.")

def get_sql_connection():
    """
    This method simply returns the sql connection based on the DB type
    and the connection settings defined in the `db.conf`.
    :return: connection
    """
    # Reading the Database configurations
    db_config = configparser.ConfigParser()
    db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
    # db_config.read(db_file)
    db_type = db_config["SYSTEM"]["db_type"]

    # (0731): newly added.
    if db_type == "MSSQL":
        server = db_config[db_type]["server"]
        database = db_config[db_type]["database"]
        driver = db_config[db_type]["driver"]

        return pyodbc.connect(
            r"Driver=" + driver + ";Server=" + server + ";Database=" + database + ";Trusted_Connection=yes;")

    elif db_type == "postgresql":
        host = db_config[db_type]["host"]

        # (1030): newly added.
        db_name = db_config[db_type]["database"]
        # if args.db_name is not None:
        #     db_name = args.db_name

        port = db_config[db_type]["port"]
        user = db_config[db_type]["user"]
        password = db_config[db_type]["password"]

        connection = psycopg2.connect(host=host, database=db_name, port=port,
                                      user=user, password=password)
        connection.autocommit = True
        connection.commit()

        return connection

    else:
        raise NotImplementedError

def get_sql_connection_v2(db_name):
    """
    This method simply returns the sql connection based on the DB type
    and the connection settings defined in the `db.conf`.
    :return: connection
    """
    # Reading the Database configurations
    db_config = configparser.ConfigParser()
    db_config.read(constants.ROOT_DIR + constants.DB_CONFIG)
    # db_config.read(db_file)
    
    host = db_config[db_name]["host"]

    # (1030): newly added.
    database = db_config[db_name]["database"]
    # if args.db_name is not None:
    #     db_name = args.db_name

    port = db_config[db_name]["port"]
    user = db_config[db_name]["user"]
    password = db_config[db_name]["password"]

    connection = psycopg2.connect(host=host, database=database, port=port,
                                user=user, password=password)
    connection.autocommit = True
    connection.commit()

    return connection


    
def close_sql_connection(connection):
    """
    Take care of the closing process of the SQL connection
    :param connection: sql_connection
    :return: operation status
    """
    return connection.close()


# run in main
if __name__ == "__main__":
    drop_connection(get_sql_connection())