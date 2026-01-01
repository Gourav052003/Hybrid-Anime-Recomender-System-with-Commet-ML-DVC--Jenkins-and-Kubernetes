import os
import pandas
from Anime.logger import get_logger
from Anime.customException import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError("File is not in the given path")
        
        with open(file_path,"r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            logger.info("Successfully read the YAML file")
            return config
        
    except Exception as e:
        logger.error("Error reading the YAML file")
        raise CustomException("Failed to read YAML file", e)
        