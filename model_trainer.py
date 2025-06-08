import os
import sys
from dataclasses import dataclass
from sklearn.metrics import f1_score,recall_score,precision_score




from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import (DecisionTreeClassifier)
from sklearn.ensemble import (AdaBoostClassifier,GradientBoostingClassifier,
                              RandomForestClassifier)

from students.exception import StudentException
from students.logging import logging

from src.utils import save_object,model_evaluate

##from src.utils2 import evaluate_models
@dataclass
class ModelTrainerConfig:
    trained_model_path:str = os.path.join("model.pkl")

class ModelTrainer:
    def __init__(self):
        try:
            self.model_trainer_config = ModelTrainerConfig()
        except Exception as e:
            raise StudentException(e,sys)
    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("splitting the data into training and testing data")
            x_train,x_test,y_train,y_test = (
                train_arr[:,:-1],
                test_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,-1],
            )
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
            params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,64,]
            }
            
              }
            report_dict:dict = model_evaluate(x_train=x_train,x_test=x_test,
                                               y_train=y_train,y_test=y_test,
                                               models=models,params=params)
            ## now i have the report dictionary
            ## so what actually this report dictionary contain in it
            ## and how actually i can use this report dictionary
            max_accuracy = max(sorted(report_dict.values()))
            best_model_name = list(report_dict.keys())[
                list(report_dict.values()).index(max_accuracy)
            ]
            best_model = models[best_model_name]

            if max_accuracy<0.6:
                raise StudentException("There is no best machine learning model we found")
            y_pred_test = best_model.predict(x_test)
            f1_score_value_test = f1_score(y_test,y_pred_test)
            y_pred_train = best_model.predict(x_train)
            f1_score_value_train = f1_score(y_train,y_pred_train)
            logging.info("Saving my best machine learning model")
            save_object(self.model_trainer_config.trained_model_path,best_model)


            return (f1_score_value_train,f1_score_value_test,self.model_trainer_config.trained_model_path)
            
        
        
        
        except Exception as e:
            raise StudentException(e,sys)