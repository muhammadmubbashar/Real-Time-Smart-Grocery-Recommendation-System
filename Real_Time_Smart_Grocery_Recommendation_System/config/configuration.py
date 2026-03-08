import os
import sys
from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException
from Real_Time_Smart_Grocery_Recommendation_System.utils.utils import read_yaml_file
from Real_Time_Smart_Grocery_Recommendation_System.entity.config_entity import DataIngestionConfig, DataValidationConfig
from Real_Time_Smart_Grocery_Recommendation_System.constants import *

class AppConfiguration:
    def __init__(self, config_file_path=CONFIG_FILE_PATH):
        try:
            self.config_info = read_yaml_file(config_file_path)
        except Exception as e:
            raise AppException(e, sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        try:
            data_ingestion_config = self.config_info['data_ingestion_config']
            artifact_dir = self.config_info['artifact_config']['artifact_dir']
            dataset_dir = data_ingestion_config['dataset_dir']
            ingested_data_dir = os.path.join(artifact_dir, dataset_dir, data_ingestion_config['ingested_dir'])
            raw_data_dir = os.path.join(artifact_dir, dataset_dir, data_ingestion_config['raw_data_dir'])
    
            
            response = DataIngestionConfig(download_dataset_url= data_ingestion_config['download_dataset_url'],  raw_data_dir=raw_data_dir, ingested_dir=ingested_data_dir)
            logging.info(f"Data Ingestion Config: {response}")
            return response
        except Exception as e:
            raise AppException(e, sys)
    
    def get_data_validation_config(self) -> DataValidationConfig:
        try:
            data_validation_config = self.config_info['data_validation_config']
            data_ingestion_config = self.config_info['data_ingestion_config']
            artifact_dir = self.config_info['artifact_config']['artifact_dir']
            dataset_dir = data_ingestion_config['dataset_dir']

            # CSV file paths after ingestion
            order_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['order_csv_file'])
            product_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['product_csv_file'])
            order_products_prior_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['order_products_prior_csv_file'])
            order_products_train_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['order_products_train_csv_file'])
            department_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['department_csv_file'])
            aisles_csv_file = os.path.join(artifact_dir, dataset_dir, data_ingestion_config["ingested_dir"], data_validation_config['aisles_csv_file'])

            # directories for clean and serialized data
            clean_data_dir = os.path.join(artifact_dir, dataset_dir, data_validation_config['clean_data_dir'])
            serialized_objects_dir = os.path.join(artifact_dir, dataset_dir, data_validation_config['serialized_objects_dir'])

            response = DataValidationConfig(
                order_csv_file=order_csv_file,
                product_csv_file=product_csv_file,
                order_products_prior_csv_file=order_products_prior_csv_file,
                order_products_train_csv_file=order_products_train_csv_file,
                department_csv_file=department_csv_file,
                aisles_csv_file=aisles_csv_file,
                clean_data_dir=clean_data_dir,
                serialized_objects_dir=serialized_objects_dir
            )

            logging.info(f"Data Validation Config: {response}")
            return response

        except Exception as e:
            raise AppException(e, sys) from e