import os
import sys
import pickle
import pandas as pd

from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException


class DataTransformation:
    def __init__(self, app_config: AppConfiguration = AppConfiguration()):
        try:
            self.data_transformation_config = app_config.get_data_transformation_config()
            self.data_validation_config = app_config.get_data_validation_config()
        except Exception as e:
            raise AppException(e, sys) from e

    def get_data_transformer(self):
        """
        Load cleaned data and engineer features for model training
        """
        try:
            logging.info("Loading cleaned data for transformation")
            
            # Load the cleaned dataset
            df = pd.read_csv(self.data_transformation_config.clean_data_file_path)
            logging.info(f"Cleaned Data Shape: {df.shape}")
            
            # Load the original data for feature engineering
            orders = pd.read_csv(self.data_validation_config.order_csv_file)
            products_prior = pd.read_csv(self.data_validation_config.order_products_prior_csv_file)
            
            logging.info("Starting feature engineering - this may take several minutes")
            
            # Feature 1: Order Gap (recency)
            user_order = products_prior.merge(
                orders[["order_id", "user_id", "order_number"]], 
                on="order_id", 
                how="left"
            )
            
            user_last_order = orders.groupby("user_id")["order_number"].max().reset_index(
                name="max_order_number"
            )
            logging.info("Calculated max_order_number")
            
            user_product_last_order = (
                user_order.groupby(["user_id", "product_id"])["order_number"]
                .max()
                .reset_index(name="last_product_order")
            )
            logging.info("Calculated last_product_order")
            
            recency = user_product_last_order.merge(user_last_order, on="user_id", how="left")
            recency["order_gap"] = recency["max_order_number"] - recency["last_product_order"]
            
            # Merge order_gap to dataset
            df = df.merge(
                recency[["user_id", "product_id", "order_gap"]], 
                on=["user_id", "product_id"], 
                how="left"
            )
            logging.info("Added order_gap feature")
            
            # Feature 2: User Total Orders
            user_total_orders = user_last_order.copy()
            user_total_orders.rename(
                columns={"max_order_number": "user_total_orders"}, 
                inplace=True
            )
            
            df = df.merge(user_total_orders, on="user_id", how="left")
            logging.info("Added user_total_orders feature")
            
            # Feature 3: Purchase Ratio
            df["purchase_ratio"] = df["purchase_count"] / df["user_total_orders"]
            logging.info("Added purchase_ratio feature")
            
            # Feature 4: Product Popularity
            product_popularity = (
                user_order.groupby("product_id")
                .size()
                .reset_index(name="product_popularity")
            )
            logging.info("Calculated product_popularity")
            
            df = df.merge(product_popularity, on="product_id", how="left")
            logging.info("Added product_popularity feature")
            
            # Handle missing values
            df = df.fillna(0)
            
            logging.info(f"Final Dataset Shape: {df.shape}")
            logging.info(f"Dataset Columns: {df.columns.tolist()}")
            
            # Save transformed dataset to CSV
            logging.info("Saving transformed data...")
            os.makedirs(self.data_transformation_config.transformed_data_dir, exist_ok=True)
            transformed_data_path = os.path.join(
                self.data_transformation_config.transformed_data_dir, 
                "transformed_data.csv"
            )
            df.to_csv(transformed_data_path, index=False)
            logging.info(f"✓ Saved transformed data CSV to {transformed_data_path}")
            
            # Save transformed data as pickle in transformed_data folder
            transformed_pkl_path = os.path.join(
                self.data_transformation_config.transformed_data_dir,
                "transformed_data.pkl"
            )
            with open(transformed_pkl_path, "wb") as f:
                pickle.dump(df, f)
            logging.info(f"✓ Saved transformed data pickle to {transformed_pkl_path}")
            
            # Save necessary pkl files in serialized_objects for model trainer
            logging.info("Saving serialized objects for model training...")
            os.makedirs(self.data_validation_config.serialized_objects_dir, exist_ok=True)
            
            # Save transformed data pickle in serialized_objects
            transformed_pkl_serialized = os.path.join(
                self.data_validation_config.serialized_objects_dir,
                "transformed_data.pkl"
            )
            with open(transformed_pkl_serialized, "wb") as f:
                pickle.dump(df, f)
            logging.info(f"✓ Saved transformed data to serialized_objects")
            
            # Save unique product IDs
            product_ids = df["product_id"].unique()
            product_ids_path = os.path.join(
                self.data_validation_config.serialized_objects_dir,
                "product_ids.pkl"
            )
            with open(product_ids_path, "wb") as f:
                pickle.dump(product_ids, f)
            logging.info(f"✓ Saved {len(product_ids)} unique product IDs")
            
            # Save unique user IDs
            user_ids = df["user_id"].unique()
            user_ids_path = os.path.join(
                self.data_validation_config.serialized_objects_dir,
                "user_ids.pkl"
            )
            with open(user_ids_path, "wb") as f:
                pickle.dump(user_ids, f)
            logging.info(f"✓ Saved {len(user_ids)} unique user IDs")
            
            return df

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise AppException(e, sys) from e

    def initiate_data_transformation(self):
        try:
            logging.info(f"{'='*20}Data Transformation log started.{'='*20}")
            self.get_data_transformer()
            logging.info(f"{'='*20}Data Transformation log completed.{'='*20}\n\n")
        except Exception as e:
            raise AppException(e, sys) from e