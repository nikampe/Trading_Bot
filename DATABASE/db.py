import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine
import pandas as pd
import numpy as np

from DATABASE.tables import TABLES

HOST = "XXXXXXXXXXXXXXXXXXXXX"
USER = "XXXXXXXXXXXXXXXXXXXXX"
PASSWORD = "XXXXXXXXXXXXXXXXXXXXX"
DB_NAME = "XXXXXXXXXXXXXXXXXXXXX"

class DB:
    def __init__(self):
        self.host = HOST
        self.user = USER
        self.password = PASSWORD
        self.database = DB_NAME
    def connect(self):
        try:
            conn = mysql.connector.connect(
                host = self.host, 
                user = self.user, 
                password = self.password,
                database = self.database
            )
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
                print("DB authentication failed.")
            elif err.errno == errorcode.ER_BAD_DB_ERROR:
                print("DB does not exist")
            else:
                print(err)
        return conn
    def createTables(self):
        conn = self.connect()
        cursor = conn.cursor()
        for table_name in TABLES:
            table_description = TABLES[table_name]
            try:
                print("Creating table {}: ".format(table_name), end = '')
                cursor.execute(table_description)
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("already exists.")
                else:
                    print(err.msg)
            else:
                print("OK")
        cursor.close()
        conn.close()
    def getRequest(self, table, columns):
        conn = self.connect()
        query = "SELECT " + columns + " FROM " + table
        response = pd.read_sql(query, conn)
        conn.close()
        return response
    def postRequest(self, df, table):
        engine = create_engine('mysql+mysqlconnector://'+ USER + ':' + PASSWORD + '@' + HOST + '/' + DB_NAME)
        conn = self.connect()
        df.to_sql(table, con = engine, if_exists = 'append', index = False)
        conn.close()

if __name__ == '__main__':
    DB().createTables()