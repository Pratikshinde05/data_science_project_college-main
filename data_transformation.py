## here i will be writting the code for creating the data transformation pipeline using my own logic
import os
import sys
import pandas as pd
import numpy as np
from src.utils import save_object
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from students.exception import StudentException
from students.logging import logging

target_column = "passed"
## creating the data transformation config
@dataclass
class DataTransformationConfig:
   processor_obj:str = os.path.join("processor.pkl")



class DataTransformation:
    def __init__(self):
      try:
         self.data_transformation_config=DataTransformationConfig()
      except Exception as e:
         raise StudentException(e,sys)
      
    def get_transformation_preprocessor(self):
      "This function will create the processor.pkl file, whcih will be used for the transformation of the data"
      try:
         numerical_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures',  'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

         categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',  'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',  'internet', 'romantic']

        



         numerical_pipeline = Pipeline(
            steps = [("imputer",SimpleImputer(strategy="median")),
                     ("standard_scaler",StandardScaler())]
         )
         
         categorical_pipeline = Pipeline(
         steps = [("one_hot_encoder",OneHotEncoder()),
                  ("imputer",SimpleImputer(strategy="most_frequent"))]  
        )
         ## now till here i have created two pipelines 
         ## one for numerical columns and other for categorical columns
         ## now i will combine bothe the pipelines 
         logging.info(f"categorical columns:{categorical_columns}")
         logging.info(f"numerical columns:{numerical_columns}")

         processor = ColumnTransformer(
            [("numerical_columns",numerical_pipeline,numerical_columns),
             ("categorical_columns",categorical_pipeline,categorical_columns)]
         )
         return processor
    


         
      except Exception as e:
         raise StudentException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
       try:
          train_data = pd.read_csv(train_path)
          train_data["passed"] = train_data["passed"].replace({"yes": 1, "no": 0})

          test_data = pd.read_csv(test_path)
          test_data["passed"] = test_data["passed"].replace({"yes": 1, "no": 0})

          logging.info("loaded training and testing data into my local system")

          input_fetaures_train_data = train_data.drop(columns = [target_column],axis=1)
          input_features_test_data = test_data.drop(columns = [target_column],axis=1)
          target_feature_train_data = train_data[target_column]
          target_feature_test_data = test_data[target_column]

          processor_obj = self.get_transformation_preprocessor()
          input_fetaures_train_arr = processor_obj.fit_transform(input_fetaures_train_data)
          input_features_test_arr = processor_obj.transform(input_features_test_data)
          logging.info("successfully transformed my train data and test data")
          train_arr = np.c_[input_fetaures_train_arr,np.array(target_feature_train_data)]
          test_arr = np.c_[input_features_test_arr,np.array(target_feature_test_data)]

          save_object(self.data_transformation_config.processor_obj,processor_obj)
          logging.info("saved processor object for transformation")

          return (train_arr,test_arr,self.data_transformation_config.processor_obj)







       except Exception as e:
          raise StudentException(e,sys)


