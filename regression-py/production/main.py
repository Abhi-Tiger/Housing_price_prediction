import mlflow
from training import train_model
from feature_engineering import transform_features
from scoring import score_model
import subprocess
import sys
import os.path as op
import pyarrow.parquet as pq
from sklearn.metrics import mean_squared_error
import numpy as np
from data_cleaning import clean_order_table,clean_product_table,clean_housing_table,create_training_datasets
# standard code-template imports
from ta_lib.core.api import (
    create_context
)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("HousePricePrediction")
    config_path = r"production/conf/config.yml"
    context = create_context(config_path)
    # print(list_datasets(context))
    with mlflow.start_run(run_name="Main Script Run") as main_run:

        with mlflow.start_run(run_name="Data_Cleaning", nested=True):
            result1 = subprocess.run([sys.executable, 'home/abhinavsharma/housing_price/regression-py/production/data_cleaning.py'], capture_output=True, text=True)
            # clean_pt = clean_product_table(cont/ext,{})
            # clean_ot = clean_order_table(context,{})
            clean_st = clean_housing_table(context,{})
            create_training_datasets(context,params = {"test_size": 0.2, "target":"median_house_value"})
            # mlflow.log_param("product_table_rows", clean_pt.shape[0])
            # mlflow.log_param("product_table_cols", clean_pt.shape[1])
            # mlflow.log_param("order_table_rows", clean_ot.shape[0])
            # mlflow.log_param("order_table_cols", clean_ot.shape[1])
            mlflow.log_param("housing_table_rows", clean_st.shape[0])
            mlflow.log_param("housing_table_cols", clean_st.shape[1])
            mlflow.log_param("train_validation_split", "80-20")



        with mlflow.start_run(run_name="Feature_Engineering", nested=True):
            result1 = subprocess.run([sys.executable, 'home/abhinavsharma/housing_price/regression-py/production/feature_engineering.py'], capture_output=True, text=True)
            transform_features(context,params = {"outliers": {"method": "mean","drop": False }})

        with mlflow.start_run(run_name="Training", nested=True):
            result1 =  subprocess.run([sys.executable, 'home/abhinavsharma/housing_price/regression-py/production/training.py'], capture_output=True, text=True)
            train_model(context,{})
            # mlflow.log_param("param_name", res)
        with mlflow.start_run(run_name="Scoring", nested=True):
            result1 = subprocess.run([sys.executable, 'home/abhinavsharma/housing_price/regression-py/production/scoring.py'], capture_output=True, text=True)
            score_model(context,{})


            # Read Parquet files
            df_target = pq.read_table('data/test/housing/target.parquet').to_pandas()
            df_actual = pq.read_table('data/test/housing/scored_output.parquet').to_pandas()
            # print(df_target.columns)
            # print(df_actual.columns)

            # Extract columns containing target and actual values
            target_column = 'median_house_value'
            actual_column = 'yhat'

            target_values = df_target[target_column]
            actual_values = df_actual[actual_column]

            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(target_values, actual_values))
            mlflow.log_metric("RMSE",rmse)