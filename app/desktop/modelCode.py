from ...src.training.training import makeModel
from ...tests.preprocess import makePreProcessedDataForFrontEndWithPadding

class Model():
    model = None
    def __init__(self, path) -> None:
        if self.model is not None:
            return
        else:
            self.model = makeModel(path)
    def predict(self, videoPaths):
        assert self.model is not None
        X = makePreProcessedDataForFrontEndWithPadding(videoPaths, 266)
        return self.model.predict(X)
            