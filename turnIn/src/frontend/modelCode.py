import pandas as pd

from ..infer import makeModel
from ...utils.preprocess import makePreProcessedDataForFrontEndWithPadding, addingPaddingToPreProcessedData
from .. import config

class Model():
    model = None
    def __init__(self, path) -> None:
        if self.model is not None:
            return
        else:
            self.model = makeModel(path)
    def predictDict(self, dataDict):
        assert self.model is not None, 'model not initialized'
        X = addingPaddingToPreProcessedData(pd.DataFrame([dataDict]), config.MAX_TOKENS) # wrap dict so it translates the sequnces as cells and not rows
        return self.model.predict(X)
    def predictFiles(self, videoPaths):
        assert self.model is not None, 'model not initialized'
        X = makePreProcessedDataForFrontEndWithPadding(videoPaths, config.MAX_TOKENS)
        return self.model.predict(X)
            