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
            
            params={
                
                "Decision_Tree": [
                    {
                        "criterion": [
                            "squared_error",
                            "friedman_mse",
                            "absolute_error",
                            "poisson",
                        ]
                    }
                ],
                
                "Random_Forest": [
                    {
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                        "max_depth": [None, 5, 10, 20],
                        "min_samples_split": [2, 5, 10],
                        "min_samples_leaf": [1, 2, 4]
                    }
                ],
                
                "Gradient_Boosting":[
                    {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    }
                ],
                
                "Linear_Regression":[
                    {}
                ],
                
                "K-Neighbors_Classifier": [
                    {
                        "n_neighbors": [3, 5, 7, 9]
                    }
                ],
                
                "XGBoosting_Classifier":[
                    {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    }
                ],
                
                "CatBoosting_Classifier":[
                    {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100],
                    }
                ],
                
                "AdaBoost_Classifier":[
                    {
                        "learning_rate": [0.1, 0.01, 0.5, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    }
                ]    
            }
            
            model_report:dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models, param=params)
            
            
            # Extract scores and models from the report
            scores = {k: v[0] for k, v in model_report.items()}
            best_model_score = max(scores.values())
            best_model_name = max(scores, key=scores.get)
            best_model = model_report[best_model_name][1]

            
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
            
        
    