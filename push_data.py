import os
import sys
import json
from dotenv import load_dotenv
from students.exception import StudentException
import pandas as pd
import numpy as np
import pymongo



load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
print(MONGO_DB_URL)

class StudentDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
           raise StudentException(e,sys)
    def csv_to_json_converter(self,data_path):
        try:
            data = pd.read_csv(data_path)
            data.reset_index(drop = True,inplace = True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise StudentException(e,sys)

    def insert_data_mongodb(self,database,collection,records):
      try:
            self.database = database
            self.collection = collection
            self.records = records

            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            self.database = self.mongo_client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            
            
      except Exception as e:
            raise StudentException(e,sys)
      

if __name__ == "__main__":
        data_path = "data\student-data.csv"
        database = "student_db"
        collection = "student_data"
        network = StudentDataExtract()
        records = network.csv_to_json_converter(data_path)
        print(records)
        network.insert_data_mongodb(database,collection,records)