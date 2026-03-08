import os
import sys
import pickle
import pandas as pd

from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException


class DataValidation:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys)

    def preprocess_data(self):
        try:
            logging.info("Starting data validation and preprocessing")

            # Load CSVs
            orders = pd.read_csv(self.data_validation_config.order_csv_file)
            products = pd.read_csv(self.data_validation_config.product_csv_file)
            aisles = pd.read_csv(self.data_validation_config.aisles_csv_file)
            products_prior = pd.read_csv(self.data_validation_config.order_products_prior_csv_file)
            products_train = pd.read_csv(self.data_validation_config.order_products_train_csv_file)
            departments = pd.read_csv(self.data_validation_config.department_csv_file)

            logging.info(f"Orders Shape: {orders.shape}")
            logging.info(f"Products Shape: {products.shape}")
            logging.info(f"Aisles Shape: {aisles.shape}")
            logging.info(f"Prior Shape: {products_prior.shape}")
            logging.info(f"Train Shape: {products_train.shape}")

            # Basic column validation
            required_cols = {
                'orders': ["order_id", "user_id", "order_number"],
                'products': ["product_id"],
                'prior': ["order_id", "product_id"],
                'train': ["order_id", "product_id"]
            }

            for col in required_cols['orders']:
                if col not in orders.columns:
                    raise Exception(f"{col} missing in orders.csv")
            for col in required_cols['products']:
                if col not in products.columns:
                    raise Exception(f"{col} missing in products.csv")
            for col in required_cols['prior']:
                if col not in products_prior.columns:
                    raise Exception(f"{col} missing in order_products__prior.csv")
            for col in required_cols['train']:
                if col not in products_train.columns:
                    raise Exception(f"{col} missing in order_products__train.csv")

            logging.info("Column validation successful")

            # Preprocessing
            user_order = products_prior.merge(
                orders[["order_id", "user_id", "order_number"]],
                on="order_id", how="left"
            )

            user_products = (
                user_order.groupby(["user_id", "product_id"])
                .size().reset_index(name="purchase_count")
            )

            products_train = products_train.merge(
                orders[["order_id", "user_id"]],
                on="order_id", how="left"
            )
            products_train["reordered"] = 1
            train = products_train[["user_id", "product_id", "reordered"]]

            dataset = user_products.merge(train, on=["user_id", "product_id"], how="left")
            dataset["reordered"] = dataset["reordered"].fillna(0)
            dataset = dataset.drop_duplicates()

            # Save cleaned dataset
            os.makedirs(self.data_validation_config.clean_data_dir, exist_ok=True)
            clean_data_path = os.path.join(self.data_validation_config.clean_data_dir, "clean_data.csv")
            dataset.to_csv(clean_data_path, index=False)
            logging.info(f"Saved cleaned data to {clean_data_path}")

            # Save serialized dataset
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            pickle_path = os.path.join(self.data_validation_config.serialized_objects_dir, "clean_data.pkl")
            with open(pickle_path, "wb") as f:
                pickle.dump(dataset, f)
            logging.info(f"Saved serialized dataset to {pickle_path}")

            return dataset

        except Exception as e:
            raise AppException(e, sys)

    def initiate_data_validation(self):
        try:
            logging.info(f"{'='*20}Data Validation log started.{'='*20}")
            self.preprocess_data()
            logging.info(f"{'='*20}Data Validation log completed.{'='*20}\n\n")
        except Exception as e:
            raise AppException(e, sys) from e