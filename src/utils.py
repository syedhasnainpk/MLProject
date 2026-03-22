import os 
import sys
import numpy as np 
import pandas as pd 
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill


import dill

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

def evaluate_model(x_train, y_train, x_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():
            # If parameters are provided for this model, use GridSearchCV
            if model_name in param and param[model_name]:
                gs = GridSearchCV(model, param[model_name], cv=3)
                gs.fit(x_train, y_train)
                best_model = gs.best_estimator_
            # If no parameters, just fit the model
            else:
                model.fit(x_train, y_train)
                best_model = model

            # Predictions
            y_train_pred = best_model.predict(x_train)
            y_test_pred = best_model.predict(x_test)

            # Scores
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = (test_model_score, best_model)

        return report
    except Exception as e:
        raise CustomException(e, sys)

    
def load_object(file_path):
        try:
            with open(file_path, 'rb') as file_obj:
                return dill.load(file_obj)
            
        except Exception as e:
            raise CustomException(e, sys)    