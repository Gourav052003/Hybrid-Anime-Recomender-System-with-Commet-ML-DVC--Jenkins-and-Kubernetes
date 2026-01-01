import os
import pandas as pd
from google.cloud import storage
from Anime.logger import get_logger
from Anime.customException import CustomException
from Config.pathsConfig import *
from Utils.commonFunctions import read_yaml
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)

class DataIngestion:

    def __init__(self,config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.bucket_file_names = self.config["bucket_file_names"]

        os.makedirs(RAW_DIR,exist_ok=True)

        logger.info("Data Ingestion Started.........")
    
    def download_csv_from_GCP(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)

            for file_name in self.bucket_file_names:
                file_path = os.path.join(RAW_DIR,file_name)
                
                if file_name=="animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    data = pd.read_csv(file_path,nrows=5000000)
                    data.to_csv(file_path,index=False)

                    logger.info("Large file detected, dowloading 5M rows only")

                else:

                    blob =  bucket.blob(file_name)
                    blob.download_to_filename(file_path)

                    logger.info("Downloading small files [anime.csv, anime_with_synopsis]")

        except Exception as e:
            logger.error("Error while Ingesting Data from GCP", e)
            raise CustomException("Failed to Ingest Data", e)
        

    def run(self):

        try:
            logger.info("Starting data Ingestion Process")
            self.download_csv_from_GCP()
            logger.info("Data Ingestion Completed")
        
        except CustomException as ce:
            logger.error(f"Custom Exception during Data Ingestion : {str(ce)}")


if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
