import os
import sys
from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
import kagglehub
import zipfile

class DataIngestion:
    def __init__(self, app_config: AppConfiguration):
        '''Initialize the DataIngestion component with the provided configuration.
        data_ingestion_config: DataIngestionConfig '''
        try:
            logging.info("Initializing Data Ingestion component...")
            self.data_ingestion_config = app_config.get_data_ingestion_config()

        except Exception as e:
            raise AppException(e, sys)
    def download_data(self):
        """
        Download dataset from Kaggle
        """
        try:
            dataset_name = self.data_ingestion_config.download_dataset_url
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(f"Downloading dataset {dataset_name} from Kaggle")

            # Download dataset
            dataset_path = kagglehub.dataset_download(dataset_name)

            logging.info(f"Dataset downloaded at {dataset_path}")

            return dataset_path

        except Exception as e:
            raise AppException(e, sys) from e

    def extract_zip_file(self, dataset_path: str):

        try:
            ingested_dir = self.data_ingestion_config.ingested_dir
            os.makedirs(ingested_dir, exist_ok=True)

            for file in os.listdir(dataset_path):

                if file.endswith(".zip"):

                    zip_file_path = os.path.join(dataset_path, file)

                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(ingested_dir)

                    logging.info(f"Extracted {zip_file_path} into {ingested_dir}")

                else:
                    # If file is already csv just copy
                    src = os.path.join(dataset_path, file)
                    dst = os.path.join(ingested_dir, file)
                    os.replace(src, dst)

        except Exception as e:
            raise AppException(e, sys) from e
    def initiate_data_ingestion(self):
        try:
            dataset_path = self.download_data()
            self.extract_zip_file(dataset_path)

            logging.info(f"{'='*20}Data Ingestion log completed.{'='*20}")

        except Exception as e:
            raise AppException(e, sys) from e