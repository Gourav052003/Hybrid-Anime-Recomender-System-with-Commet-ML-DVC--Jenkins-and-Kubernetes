from Config.pathsConfig import *
from Utils.commonFunctions import read_yaml
from Anime.dataProcessing import DataProcessor
from Anime.modelTraining import ModelTraining
from Anime.dataIngestion import DataIngestion

if __name__=="__main__":
    
    # data_ingestor = DataIngestion(read_yaml(CONFIG_PATH))
    # data_ingestor.run()
  
    data_processor = DataProcessor(ANIMELIST_CSV,PROCESSED_DIR)
    data_processor.run()

    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
