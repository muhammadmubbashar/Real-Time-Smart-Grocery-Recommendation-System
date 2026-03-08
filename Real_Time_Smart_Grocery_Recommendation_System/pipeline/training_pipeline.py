from Real_Time_Smart_Grocery_Recommendation_System.components.data_ingestion import DataIngestion
from Real_Time_Smart_Grocery_Recommendation_System.components.data_validation import DataValidation
from Real_Time_Smart_Grocery_Recommendation_System.components.data_transformation import DataTransformation
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException
import sys


class TrainingPipeline:
    def __init__(self):
        self.app_config = AppConfiguration()
        self.data_ingestion = DataIngestion(self.app_config)
        self.data_validation = DataValidation(self.app_config)
        self.data_transformation = DataTransformation(self.app_config)
      
    def start_data_ingestion(self):
        """
        Starts the data ingestion pipeline
        :return: none
        """
        try:
            logging.info("Starting Data Ingestion pipeline...")
            self.data_ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed successfully")
        except Exception as e:
            raise AppException(e, sys) from e

    def start_data_validation(self):
        """
        Starts the data validation pipeline
        :return: none
        """
        try:
            logging.info("Starting Data Validation pipeline...")
            self.data_validation.initiate_data_validation()
            logging.info("Data Validation completed successfully")
        except Exception as e:
            raise AppException(e, sys) from e

    def start_data_transformation(self):
        """
        Starts the data transformation pipeline
        :return: none
        """
        try:
            logging.info("Starting Data Transformation pipeline...")
            self.data_transformation.initiate_data_transformation()
            logging.info("Data Transformation completed successfully")
        except Exception as e:
            raise AppException(e, sys) from e

    def run_pipeline(self):
        """
        Run the complete training pipeline
        """
        try:
            logging.info(f"{'='*50}Training Pipeline Started{'='*50}\n")
            
            # Step 1: Data Ingestion
            self.start_data_ingestion()
            
            # Step 2: Data Validation
            self.start_data_validation()
            
            # Step 3: Data Transformation
            self.start_data_transformation()
            
            logging.info(f"{'='*50}Training Pipeline Completed{'='*50}\n")
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise AppException(e, sys) from e


if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        print("\n" + "="*60)
        print("Pipeline completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()