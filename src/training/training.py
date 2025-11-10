#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import torchmetrics
from torchmetrics import Metric

import lightning as L
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd

import numpy as np

import ast


# # postitional encoding

# In[ ]:


class PositionEncoding(nn.Module):
    """
    This code is the PositionEncoding part of stat quests transformer decoder video
    https://www.youtube.com/watch?v=C9QSpl5nmrY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 
    With adjustments and corrections for our model
    
    We replaced the comments he had with my own though so we could understand whats going on better.
    We would've like to use a built in module for the positional encoding part of the transformer were
    making but we couldn't find a pre built module from pytorch
    
    Instead of computing the positial encodings each time, we can make a lookup table
    because we know the max length of tokens aka the max number of posititons 
    
    these are the equations for positional encoding 
    
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)
    
    
    So this class when initialized save all of the calculation then when you call forward it just does a lookup
    """
    def __init__(self, d_model, max_len):
        """
        d_model should be the same as the number of embeddings.
        max_len should be the max number of tokens processed at a time
        """
        super().__init__() #calls nn.Module.__init__() which starts building computation graph in background
        
        self.d_model=d_model
        
        isOdd = False
        #d_model needs to be even so self.pe[:, 1::2] works later 
        if d_model % 2 != 0:
            isOdd = True
            d_model += 1
                
        #initializing lookup table
        pe = torch.zeros(max_len, d_model)
        
        #make list of possible positions then make them a column matrix for further math 
        # [
        # [0],
        # [1],
        # [2],
        # ...,
        # [end]
        # ]
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        
        #handles indexing for alternation between sin and cos because positional encoding splits the embeddings into 
        # pairs, 
        # essentially meaning even pos use
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # and odd pos use
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        #makes denominator term from the equation
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        
        #save formulas using the respective indexes
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        if isOdd: # we need to get rid of the extra computation at the end of the list so the dimension matches and we can perform tensor addition
            pe = pe[:, 0:-1]
        pass
        
        # #this joins the pre computed stuff with the models, so if the model is on the gpu so is this stuff
        self.register_buffer('pe', pe)
        
    def forward(self, embeddings):
        """
        forward is what happens when you call the object.
        
        this method just looks up the precalculated positional encodings
        """
        return embeddings + self.pe[:, :self.d_model] # this needed to be flipped for oue model
                                                                                
        
        


# # Attention

# In[ ]:


class Attention(nn.Module): 
    """
    We originally planned on using this attention class from stat quest vide
    https://www.youtube.com/watch?v=C9QSpl5nmrY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 
    to code the attention part of the Encoder model, however, after some more research
    we figured out the pytorch already has an attention module that works and is optimized
    """
    
    def __init__(self, d_model): #d_model=num_pos_encodings=num_embeddings=(numFeatures-4)/4 + 4
        """
        This method just creates the paramaters for the query, key, and value
        """
        
        super().__init__()
        
        self.d_model=d_model
        
        # notice all of the weight matrixes have square aka n by n dimensions 
        # thats so the dimension of the model, aka d_model, doesn't change throughout the model
        self.W_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model)
        
        # when using batching dimension can look like (batch, row, col)
        # the following constants help us transpose row and column easier
        # for example instead of transpose((batch, row, col))=(col,row,batch)
        # we get transpose((batch, row, col), dim0=row, dim1=col)=(batch,col,row)
        # using negative indexes works because row should be second before last and
        # col should be the last 
        self.row_dim = -2
        self.col_dim = -1

        
    def forward(self, encodings_for_q, encodings_for_k, encodings_for_v, mask=None):
        
        #Calculate encodings for tokens
        q = self.W_q(encodings_for_q)
        k = self.W_k(encodings_for_k)
        v = self.W_v(encodings_for_v)

        ### ATTENTION(Q,K,V)=SOFTMAX((q @ k^T)/sqrt(d_model) + M) @ V
            
        ##(q @ k^T)/sqrt(d_model)
        # transpose is making sure to transpose the rows and cols while leaving the batch dim the same
        # attention score calculates similarity(aka sims) using dot product via matrix multiplication 
        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(self.d_model**0.5)

        ##(q * k^T)/sqrt(d_model)+M
        if mask is not None:
            # used for masking if we have padding or if we use masked self attention
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)
        
        ##SOFTMAX((q @ k^T)/sqrt(d_model) + M)
        #Apply softmax to get the percent of each token's value to use in attention value
        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        ##SOFTMAX((q @ k^T)/sqrt(d_model) + M) @ V
        attention_scores = torch.matmul(attention_percents, v)
        
        return attention_scores


# # FeedForwardNeuralNetwork

# In[ ]:


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_size:int, out_size:int, hidden_size:int|None=None, device=None):
        super().__init__()
        
        if hidden_size is None:
            hidden_size = in_size
            self.model = nn.Identity()
        else:
            # debating making another field for the class that essentially allows us to repeat the hidden layer n times 
            self.model = nn.Sequential(
                nn.Linear(in_size, hidden_size,device=device),
                nn.ReLU(),
                nn.Linear(hidden_size, out_features=out_size,device=device)
            )
    def forward(self, x):
        return self.model(x)
    


# # Encoder

# In[ ]:


class Encoder(L.LightningModule):
    
    def __init__(self,input_features:int, output_features:int,max_tokens:int, embed_dim:int|None=None, 
                 hidden_size:int|None=None, num_heads:int=1, batch_first:bool=True, dropOut:float=0., bias:bool=False,
                 device=None):
        """
        Note: num_heads needs to divide embed_dim evenly in order to properly split up computation
        for parallelization
        
        If batch_first = True the input is expected to be of dimension (batch, seq, feature)
        
        If hidden_size remains None then there will be no FFN.
        """
        super().__init__()
        
        if embed_dim is None:
            embed_dim = input_features
        
        ##############
        #using embedding to convolute input
        ##############
        
        self.input_embeder = nn.Linear(input_features, embed_dim, bias=bias,device=device)
        
        ##############
        #positional encoding
        ##############
        
        self.pos_encoder = PositionEncoding(embed_dim, max_len=max_tokens)
        
        ##############
        # NOTE in the paper, 'all you need is attention'
        # the multi headed attention step and the 
        # feed forward network step are repeated n time
        # so that is a possible area of architecture improvement
        ##############
        
        ##############
        #multi headed attention
        ##############
        
        assert embed_dim % num_heads == 0, 'embedding dimension: %d is not divisible by number of attention heads: %d'%(embed_dim,num_heads)
        assert 0 <= dropOut and dropOut <= 1, 'dropout: %.2f is outside of possible probability 0 <= dropout <= 1'%(dropOut)
        
        #MHAL stands for multithreaded attention layer
        self.MHA = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                           batch_first=batch_first, bias=bias, device=device)
        
        #normLayer
        self.normL_MHA = nn.LayerNorm(embed_dim, eps=0.0001, bias=True, elementwise_affine=True, device=device)
        
        ##############
        # feed forward Network
        ##############
        
        self.FFN = FeedForwardNetwork(in_size=embed_dim, hidden_size=hidden_size, out_size=embed_dim, device=device)
        
        #normLayer
        self.normL_FFN = nn.LayerNorm(embed_dim, eps=0.0001, bias=True, elementwise_affine=True, device=device)
        
        ###############
        # output layer
        ###############
        
        self.outputLayer = nn.Sequential(
            nn.Linear(in_features=embed_dim * max_tokens, out_features=output_features, bias=bias, device=device),
            nn.Softmax(dim=-1)
        )
        
        ###############
        # how is loss calculated
        ###############
        
        self.loss = nn.CrossEntropyLoss()
        
        ###############
        # metrics
        ###############
        
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=output_features)
        self.recall = torchmetrics.Recall(task='multiclass', num_classes=output_features)
        self.precision = torchmetrics.Precision(task='multiclass', num_classes=output_features)
        self.f1 = torchmetrics.F1Score(task='multiclass', num_classes=output_features)
        
    def forward(self,x,attn_mask=None,):
        """
        Method shows how data flows through the model
        
        This method also builds the nodes in the computation graph, 
        the computation graph takes care of tracking derivatives for backpropigation
        """
        embeddings = self.input_embeder(x)
        pos_encodings = self.pos_encoder(embeddings)
        residual:torch.Tensor = embeddings + pos_encodings
        
        attention_output, attention_weights = self.MHA.forward(residual, residual, residual, attn_mask=attn_mask,
                         average_attn_weights=False, need_weights=False)
        residual = self.normL_MHA(attention_output+residual)
        
        ffn_output = self.FFN(residual)
        residual = self.normL_FFN(ffn_output+residual)
        
        residual = residual.flatten(1,2)
        
        output = self.outputLayer(residual)
        
        return output
    
    def configure_optimizers(self):
        pass
        """
        (Ada)ptive (m)oment estimation, 
        like stochastic gradient decent except instead of using a fixed learning rate for all params
        it uses an adapted learning for each param
        """
        return Adam(self.parameters(), lr=0.0001)
    
    def training_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output:torch.Tensor = self.forward(input_tokens)
        loss = self.loss(output, labels)
                
        self.log_dict({'train_loss': loss, 'train_accuracy':self.accuracy(output, labels), 'train_recall':self.recall(output, labels),
                       'train_precision':self.precision(output, labels), 'train_f1':self.f1(output, labels)},
                      on_step=True,
                      on_epoch=True
                      )
                    
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens)
        loss = self.loss(output, labels)
                
        self.log_dict({'val_loss': loss, 'val_accuracy':self.accuracy(output, labels), 'val_recall':self.recall(output, labels),
                       'val_precision':self.precision(output, labels), 'val_f1':self.f1(output, labels)},
                      on_step=False,
                      on_epoch=True
                      )
                    
        return loss
    
    def test_step(self, batch, batch_idx):
        input_tokens, labels = batch
        output = self.forward(input_tokens)
        loss = self.loss(output, labels)
                
        self.log_dict({'test_loss': loss, 'test_accuracy':self.accuracy(output, labels), 'test_recall':self.recall(output, labels),
                       'test_precision':self.precision(output, labels), 'test_f1':self.f1(output, labels)},
                      on_step=False,
                      on_epoch=True
                      )
                    
        return loss


# # dataSet for data loader

# In[ ]:


class ASLDataset(Dataset):
    """
    This class helps us lazily load the data so memory doesn't blow up
    """
    def __init__(self, landmarkFile, oheFile, len, pdCacheSize, device):
        self.landmarkFile = landmarkFile
        self.oheFile = oheFile
        self.len = len
        self.device = device
        
        #This is trying to implement a sort of cache using pandas, that way
        # we don't have to read from disc 
        self.pdCache:dict[str, pd.DataFrame] = {'landmarkDF':pd.DataFrame(), 'oheDF':pd.DataFrame()}
        self.pdCacheSize= pdCacheSize
        
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
        
        X = X.values.tolist()#turn into list 
        y = y.values.tolist()#turn into list
        
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.float32, device=self.device)
        
        X = X.permute(1,0) #dimensions are flipped 
        
        return X, y


# # Running Training code

# In[ ]:


longest_num_of_frames = 266
"""
Pose landmarks: 33 landmarks × (x,y,z,present) = 132 columns
Included pose landmarks: 23 landmarks - 10 landmarks = 23 landmarks x (x,y,z,present) = 92 columns
Pose world landmarks: 33 landmarks × (x,y,z,present) = 132 columns
Included pose world landmarks: 23 landmarks - 10 landmarks = 23 landmarks x (x,y,z,present) = 92 columns
Included Left hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
Included Right hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
Overall presence indicators: 4 columns (poseLandmarks_present, poseWorldLandmarks_present, leftHandLandmarks_present, rightHandLandmarks_present)
excluded columns: idx, Video file, Gloss
included columns: padding
Total: 92 + 92 + 84 + 84 + 4 + 1 + 1 + 1 + 1 = 360 columns
Total input = 360 - 3 = 357 input columns
357 actual feature columns cause index, video file, label
((TotalInput - OverallPresenceIndicators)/5) + OverallPresenceIndicators = ((357-5)/4) + 5 = 93 embeddings
"""
n_embedings = 93 # divisible by 1, 3, 31, 93
hiddenSize = n_embedings * 2
numberOfHeads = 3 # 93 % 3 == 0 is true
dropOutPercent = .1
device = "cuda" if torch.cuda.is_available() else "cpu"



model = Encoder(input_features=357, output_features=172, embed_dim=n_embedings, max_tokens=longest_num_of_frames,
                hidden_size=hiddenSize, num_heads=numberOfHeads, batch_first=True, dropOut=dropOutPercent,
                bias=False, device=None)


landmarkFile = "../data/small_padd_training.csv"
oheFile = "../data/small_training_encoding.csv"
numSamples = 20
pandasCacheSize = 10
dataset = ASLDataset(landmarkFile, oheFile,numSamples,pandasCacheSize, device=device)

batchSize = 5
dataLoader = DataLoader(dataset=dataset, batch_size=batchSize,shuffle=True)

logger = TensorBoardLogger('tb_log', 'model_V0')

trainer = L.Trainer(logger=logger,max_epochs=100, accelerator='cpu', devices=1)
trainer.fit(model, dataLoader)

