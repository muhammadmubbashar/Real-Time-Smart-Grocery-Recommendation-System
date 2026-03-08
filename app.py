from Real_Time_Smart_Grocery_Recommendation_System.logger.logger import logging   
from Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler import AppException
import sys
logging.info("Starting the application...")

try:
    a=1/0
except Exception as e:
    logging.error("An error occurred: %s", e)
    raise AppException(e, sys)