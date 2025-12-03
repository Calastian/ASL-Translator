# %%
from torch.utils.data import Dataset
import pandas as pd
import ast
import torch

# %%
class ASLDataset(Dataset):
    """
    This class helps us lazily load the data so memory doesn't blow up
    """
    def __init__(self, landmarkFile, oheFile, len, pdCacheSize: int | None = None, device = None):
        self.landmarkFile = landmarkFile
        self.oheFile = oheFile
        self.len = len
        self.device = device
        self.pdCacheSize = pdCacheSize
        
        
        if pdCacheSize is not None:
            #This is trying to implement a sort of cache using pandas, that way
            # we don't have to read from disc 
            self.pdCache:dict[str, pd.DataFrame] | None = {'landmarkDF':pd.DataFrame(), 'oheDF':pd.DataFrame()}
        else:
            self.pdCache = None
            
        self.colsToDrop:dict = {'landmarkDF':['Video file', 'Gloss'], 'oheDF':['Video file']}
        
        def getConverters(path:str, columnsToDrop:list[str]=[])->dict:
            df_columns = pd.read_csv(path, nrows=0).columns.to_list() #get columns
            for col in columnsToDrop:
                df_columns.remove(col)
            return {col : ast.literal_eval for col in df_columns}
        
        self.pdConverters:dict[str, dict] = {'landmarkDF': getConverters(self.landmarkFile, self.colsToDrop['landmarkDF']),
                                             'oheDF': getConverters(self.oheFile, self.colsToDrop['oheDF'])}
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.pdCacheSize is not None:
            assert self.pdCache is not None, 'cache size is not none but pd cache is still not defined' # this should be true but this line helps the linter not panic
            
            # if index is not in cache load a new block of cache starting with index
            if index not in self.pdCache['landmarkDF'].index:
                self.pdCache['landmarkDF'] = pd.read_csv(self.landmarkFile, 
                                                         skiprows=range(1,index+1),
                                                         nrows=self.pdCacheSize,
                                                         converters=self.pdConverters['landmarkDF'],
                                                         index_col=0)
                                                            # skiprows=range(1,index+1) because the first line of the csv is the header line
                                                            # nrows we read should be the size of the cache
                                                            # converters come from intialization of the data frame and are used to correctly convert the csv string into their right data types
                                                            # index_col=0 tells read_csv that we want the 0th column to be the index column this prevents pandas from making its own index starting from 0 again
                self.pdCache['oheDF'] = pd.read_csv(self.oheFile, skiprows=range(1,index+1), nrows=self.pdCacheSize, converters=self.pdConverters['oheDF'], index_col=0)

            X = self.pdCache['landmarkDF'].drop(columns=self.colsToDrop['landmarkDF']).loc[index] 
            y = self.pdCache['oheDF'].drop(columns=self.colsToDrop['oheDF']).loc[index]
        else:
            X = pd.read_csv(self.landmarkFile, skiprows=range(1,index+1), nrows=1, converters=self.pdConverters['landmarkDF'], index_col=0)
            y = pd.read_csv(self.oheFile, skiprows=range(1,index+1), nrows=1, converters=self.pdConverters['oheDF'], index_col=0)

            X = X.drop(columns=self.colsToDrop['landmarkDF']).loc[index] 
            y = y.drop(columns=self.colsToDrop['oheDF']).loc[index]
         
        X = X.values.tolist()#turn into list 
        y = y.values.tolist()#turn into list
        
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        # y = torch.argmax(y).long()
        
        X = X.permute(1,0) #dimensions are flipped 
        
        return X, y


class PreloadedASLDataset(Dataset):
    """
        This class helps us lazily load the data so memory doesn't blow up
    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, landmarkFile, oheFile, length, device=None):
        self.landmarkFile = landmarkFile
        self.oheFile = oheFile
        self.length = length
        
        #From ASLDataset class ^^ Stays the same for every dataset..
        self.colsToDrop:dict = {'landmarkDF':['Video file', 'Gloss'], 'oheDF':['Video file']}
        
        def getConverters(path:str, columnsToDrop:list[str]=[])->dict:
            df_columns = pd.read_csv(path, nrows=0).columns.to_list() #get columns
            for col in columnsToDrop:
                df_columns.remove(col)
            return {col : ast.literal_eval for col in df_columns}
        
        self.pdConverters:dict[str, dict] = {'landmarkDF': getConverters(self.landmarkFile, self.colsToDrop['landmarkDF']),
                                             'oheDF': getConverters(self.oheFile, self.colsToDrop['oheDF'])}
        #END
        
        print("Loading entire dataset into memory...")
        
        # Load all data once
        landmark_df = pd.read_csv(self.landmarkFile, converters=self.pdConverters['landmarkDF'], nrows=length, index_col=0)
        ohe_df = pd.read_csv(self.oheFile, converters=self.pdConverters['oheDF'], nrows=length, index_col=0)

        # Convert to tensors once
        self.features = []
        self.labels = []
            
        for index in landmark_df.index:  # Use actual index instead of range
            # Process and convert to tensors (same as ASLDataset)
            X = landmark_df.drop(columns=self.colsToDrop['landmarkDF']).loc[index] 
            y = ohe_df.drop(columns=self.colsToDrop['oheDF']).loc[index]
            
            # Convert to lists (same as ASLDataset)
            X = X.values.tolist()
            y = y.values.tolist()
            
            # Create tensors (same as ASLDataset)
            X = torch.tensor(X, dtype=torch.float32, device=device)
            y = torch.tensor(y, dtype=torch.float32, device=device)
            y = torch.argmax(y).long()  # Convert one-hot to class index (REQUIRED for CrossEntropyLoss)
            
            X = X.permute(1,0)  # Flip dimensions (same as ASLDataset)
            
            self.features.append(X)
            self.labels.append(y)
            
            if len(self.features) % 100 == 0:
                print(f"Loaded {len(self.features)}/{len(landmark_df)} samples")
        
    def __len__(self):
        return self.length
               
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]  # Already on GPU!

# %%