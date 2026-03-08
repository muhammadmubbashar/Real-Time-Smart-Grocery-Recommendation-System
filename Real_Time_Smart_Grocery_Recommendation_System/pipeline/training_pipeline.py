from Real_Time_Smart_Grocery_Recommendation_System.components.data_ingestion import DataIngestion
from Real_Time_Smart_Grocery_Recommendation_System.components.data_validation import DataValidation
from Real_Time_Smart_Grocery_Recommendation_System.config.configuration import AppConfiguration
class TrainingPipeline:
    def __init__(self):
        self.app_config = AppConfiguration()
        self.data_ingestion = DataIngestion(self.app_config)
        self.data_validation = DataValidation(self.app_config)

    def start_data_ingestion(self):
        """
        Starts the training pipeline
        :return: none
        """
        self.data_ingestion.initiate_data_ingestion()