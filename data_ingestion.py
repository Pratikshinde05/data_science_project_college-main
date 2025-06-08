import os 
import sys
from  students.exception import StudentException
from students.logging import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from data_transformation import DataTransformationConfig,DataTransformation
from model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            ## first i will upload my data
            logging.info("Entered into data ingestion pipeline")
            df = pd.read_csv("data\student-data.csv")
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)
            logging.info("splitting the data into train and test")
            train,test = train_test_split(df,test_size=0.2,random_state=42)
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)
            train.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok=True)
            test.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion pipeline completed")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise StudentException(e,sys)

if __name__ =="__main__":
    obj = DataIngestion()
    train_path,test_path = obj.initiate_data_ingestion()
    obj2 =  DataTransformation()
    train_arr,test_arr,_ = obj2.initiate_data_transformation(train_path=train_path,test_path=test_path)
    obj3 = ModelTrainer()
    f1_score_train,f1_score_test,_ = obj3.initiate_model_trainer(train_arr=train_arr,test_arr=test_arr)
    print("train",f1_score_train)
    print("test",f1_score_test)

  
    