from Config.pathsConfig import *
from Utils.commonFunctions import read_yaml
from Anime.dataProcessing import DataProcessor
from Anime.modelTraining import ModelTraining

if __name__=="__main__":
  
    data_processor = DataProcessor(ANIMELIST_CSV,PROCESSED_DIR)
    data_processor.run()

    model_trainer = ModelTraining(PROCESSED_DIR)
    model_trainer.train_model()
