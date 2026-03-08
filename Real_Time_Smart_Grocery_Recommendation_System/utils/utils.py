import yaml
import sys  
import Real_Time_Smart_Grocery_Recommendation_System.exception.exxception_handler as ex

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            return content
    except Exception as e:
        raise ex.AppException(e, sys)