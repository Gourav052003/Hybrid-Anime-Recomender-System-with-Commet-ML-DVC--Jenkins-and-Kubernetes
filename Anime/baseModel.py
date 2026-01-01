from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Embedding, Dot, Flatten, Dense, Activation, BatchNormalization
from Utils.commonFunctions import read_yaml
from Anime.logger import get_logger
from Anime.customException import CustomException

logger = get_logger(__name__)

class BaseModel:

    def __init__(self,config_path):

        try:
            self.config = read_yaml(config_path)

            logger.info("Successfully loaded config.yaml")

        except Exception as e:
            raise CustomException("error loading Config.yaml",e)
        

    def recommenderNET(self,n_users,n_animes):

        try:
            
            embedding_size = self.config["model"]["embedding_size"]

            user = Input(name="user",shape=[1])
            user_embedding = Embedding(name="user_embedding", input_dim=n_users,output_dim=embedding_size)(user)

            anime = Input(name="anime",shape=[1])
            anime_embedding = Embedding(name="anime_embedding", input_dim=n_animes,output_dim=embedding_size)(anime)

            x = Dot(name="dot_product",normalize=True,axes=2)([user_embedding,anime_embedding])
            x = Flatten()(x)  
            x = Dense(1,kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)

            model = Model(inputs=[user,anime],outputs = x)
            model.compile(
                loss=self.config["model"]["loss"],
                metrics=self.config["model"]["metrics"],
                optimizer = self.config["model"]["optimizer"]
            )

            logger.info("model created successfully")

            return model

        except Exception as e:
            logger.erro(f"Error occured during model creation {e}")
            raise CustomException("Error during model creation",e)

      





