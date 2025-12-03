import torch
import pandas as pd
import ast

#our imports
from .architecture import Encoder
from . import config

def makeModel(modelFile):
    """Creates an instance of the ASL model

    Args:
        modelFile (hyperparameters): the ASL model hyperparameters

    Returns:
        model:the ASL model
    """
    model = Encoder(input_features=config.N_INPUTS, output_features=config.N_OUTPUTS,
                    embed_dim=config.N_EMBEDDINGS, max_tokens=config.MAX_TOKENS,
                    hidden_size=config.HIDDEN_SIZE, num_heads=config.N_HEADS,
                    batch_first=True, bias=config.BIAS, device=config.DEVICE)
    checkpoint = torch.load(modelFile, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    # model = Encoder.load_from_checkpoint('../models/ASL_Model_-v9.ckpt',map_location=torch.device('cpu'), input_features=357 , output_features=172, max_tokens=266, embed_dim=93, hiddenSize=93*2, numberOfHeads=3) # tried to lightning to load instead of the above three lines but we couldn't get it figured out

    return model

if __name__ == '__main__':
    
    def getConverters(path:str, columnsToDrop:list[str]=[])->dict:
        """helper function used to get the converter dict to turn '[1,2,3]', and '[1.0, 1.1, 1.2]'
        as a sequence of {1:list[int], 2:list[float]} instead of {1:str, 2:str} because of pandas"""
        df_columns = pd.read_csv(path, nrows=0).columns.to_list() #get columns
        for col in columnsToDrop:
            df_columns.remove(col)
        return {col : ast.literal_eval for col in df_columns} #ast.literal_eval is the actual converter being use for each col

    #read data
    x = pd.read_csv(config.INPUT_FILE, nrows=config.N_PREDICTIONS, index_col=0, #gets rid of index for input
                    converters=getConverters(config.INPUT_FILE, columnsToDrop=config.COLS_TO_DROP))
    x = x.drop(columns=config.COLS_TO_DROP)

    model = makeModel(config.MODEL_FILE)
    print(model.predict(x))