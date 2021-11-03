import pandas as pd
import mysql.connector
from mysql.connector import errorcode
from sqlalchemy import create_engine
from pymongo import MongoClient
import pymongo

from DATABASE.tables import TABLES
from DATABASE.collections import COLLECTIONS

# MySQL
HOST = "XXXXXXXXXXXX"
USER = "XXXXXXXXXXXX"
PASSWORD = "XXXXXXXXXXXX"
DB_NAME = "XXXXXXXXXXXX"

# MongoDB
CONNECTION_STRING = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

class MySQL_DB:
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
    def getLimitedRequest(self, table, columns, sort_column, limit):
        conn = self.connect()
        query = f"SELECT {columns} FROM {table} ORDER BY {sort_column} DESC LIMIT {limit};"
        response = pd.read_sql(query, conn)
        conn.close()
        return response
    def postRequest(self, df, table):
        engine = create_engine('mysql+mysqlconnector://'+ USER + ':' + PASSWORD + '@' + HOST + '/' + DB_NAME)
        conn = self.connect()
        df.to_sql(table, con = engine, if_exists = 'append', index = False)
        conn.close()

class Mongo_DB:
    def __init__(self):
        self.conn = CONNECTION_STRING
        self.connections = COLLECTIONS
    def get_database(self):
        client = MongoClient(self.conn)
        db = client['Trading_Bot']
        return db
    def createCollections(self):
        db = self.get_database()
        existing_collections = db.list_collection_names()
        print(existing_collections)
        for collection in self.connections:
            if collection in existing_collections:
                print(f'Collection {collection} already exists.')
            else:
                try:
                    db.create_collection(name = collection)
                    print(f'Collection {collection} successfully created.')
                except:
                    print('Error.')
    def getRequest(self):
        pass
    def postRequest(self, data, collection):
        db = self.get_database()
        db[collection].insert_one(data)

if __name__ == '__main__':
    MySQL_DB().createTables()
    Mongo_DB().createCollections()