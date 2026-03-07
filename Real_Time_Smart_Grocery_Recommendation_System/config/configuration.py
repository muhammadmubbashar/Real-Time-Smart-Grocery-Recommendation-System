import os
import sys
from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException
from Real_Time_Smart_Grocery_Recommendation_System.utils.utils import read_yaml_file
from Real_Time_Smart_Grocery_Recommendation_System.entity.config_entity import DataIngestionConfig
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