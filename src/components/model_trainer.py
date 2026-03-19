'''A model trainer is a component (script, module, or system) that trains a machine learning model using prepared data.'''

import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],     # Features for training: all rows, all columns except last
                train_array[:,-1],      # Labels for training: all rows, only the last column
                test_array[:, :-1],     # Features for testing: all rows, all columns except last
                test_array[:,-1]        # Labels for testing: all rows, only the last column
            )
            
            models = {
                "Random_Forest": RandomForestRegressor(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Gradient_Boosting": GradientBoostingRegressor(),
                "Linear_Regression": LinearRegression(),
                "K-Neighbors_Classifier": KNeighborsRegressor(),
                "XGBoosting_Classifier": XGBRegressor(),
                "CatBoosting_Classifier": CatBoostRegressor(),
                "AdaBoost_Classifier": AdaBoostRegressor()
            }
            
            model_report:dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            
            
            # To get the best model score from dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # To get best model name from dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            
            logging.info("Best found model on both training and testing dataset")
            
            save_obj(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )
            
            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)
            
        
    