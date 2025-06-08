import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from students.exception import StudentException
from sklearn.metrics import r2_score,f1_score
from sklearn.model_selection import GridSearchCV
def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise StudentException(e,sys)
    
def model_evaluate(x_train,y_train,x_test,y_test,models,params):
    report_dict = {}
    try:
        for i in range(len(list(models.keys()))): ## iterating through each model and also to its hyperparameters
            model = list(models.values())[i] ## got the model
            para = params[list(models.keys())[i]] ## got the hyperparameters of the model
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)
            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train) ## training of the model on the provided dataset
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            
            f1_score_train = f1_score(y_train,y_train_pred)
            f1_score_test = f1_score(y_test,y_test_pred)
            
            report_dict[list(models.keys())[i]] = f1_score_test
        return report_dict




    except Exception as e:
        raise StudentException(e,sys)

def load_object(path):
    try:
        with open(path,'rb') as file:
          return  pickle.load(file)
    except Exception as e:
        raise StudentException(e,sys)
    

import pandas as pd
import os
import pickle

def json_to_df(json_data):
    """
    Converts JSON data into a Pandas DataFrame.
    """
    try:
        df = pd.DataFrame(json_data)
        return df
    except Exception as e:
        raise ValueError(f"Error converting JSON to DataFrame: {str(e)}")


