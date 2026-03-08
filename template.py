import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')
project_name = "Real-Time-Smart-Grocery-Recommendation-System"
list_of_files = [

f"{project_name}/__init__.py",
f"{project_name}/components/__init__.py",  
f"{project_name}/components/data_ingestion.py",
f"{project_name}/components/data_validation.py",
f"{project_name}/components/data_transformation.py",
f"{project_name}/components/model_trainer.py",
f"{project_name}/config/__init__.py",
f"{project_name}/config/configuration.py",
f'{project_name}/constants/__init__.py',
f'{project_name}/entity/__init__.py',
f'{project_name}/entity/config_entity.py',
f"{project_name}/exception/__init__.py",
f"{project_name}/exception/exxception_handler.py",
f"{project_name}/logger/__init__.py",
f"{project_name}/logger/logger.py",
f"{project_name}/pipeline/__init__.py",
f"{project_name}/pipeline/training_pipeline.py",
f"{project_name}/utils/__init__.py",
f"{project_name}/utils/utils.py",
"config/config.yaml",
".dockerignore",
"Dockerfile",
"app.py",
"setup.py",

    
]
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")
    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}") 