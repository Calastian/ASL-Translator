#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

import lightning as L

import pandas as pd




# # postitional encoding

# In[ ]:


class PositionEncoding(nn.Module):
    """
    This code is the PositionEncoding part of stat quests transformer decoder video
    https://www.youtube.com/watch?v=C9QSpl5nmrY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 
    
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
    def __init__(self, d_model=112, max_len=242): #d_model=num_embeddings=(numFeatures-4)/4 + 4, max_len = max_numFrame * 2
        """
        d_model should be the same as the number of embeddings.
        max_len should be the max number of tokens processed at a time
        """
        super().__init__() #calls nn.Module.__init__() which starts building computation graph in background
        
        self.d_model=d_model
        
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
        
        #this joins the pre computed stuff with the models, so if the model is on the gpu so is this stuff
        self.register_buffer('pe', pe)
        
    def forward(self, embeddings):
        """
        forward is what happens when you call the object.
        
        this method just looks up the precalculated positional encodings
        """
        
        return embeddings + self.pe[:self.d_model, :]                                                                       ## 
        
        


# # Attention

# In[ ]:


class Attention(nn.Module): 
    """
    We originally planned on using this attention class from stat quest vide
    https://www.youtube.com/watch?v=C9QSpl5nmrY&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1 
    to code the attention part of the Encoder model, however, after some more research
    we figured out the pytorch already has an attention module that works and is optimized
    """
    
    def __init__(self, d_model=112): #d_model=num_pos_encodings=num_embeddings=(numFeatures-4)/4 + 4
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
    
    def __init__(self,input_features:int=436, output_features:int=172,embed_dim:int=112, max_tokens:int=664*2,
                 hidden_size:int|None=None, num_heads:int=1, batch_first:bool=True, dropOut:float=0., bias:bool=False,
                 device=None):
        """
        Note: num_heads needs to divide embed_dim evenly in order to properly split up computation
        for parallelization
        
        If batch_first = True the input is expected to be of dimension (batch, seq, feature)
        
        If hidden_size remains None then there will be no FFN.
        """
        super().__init__()
        
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
        self.normL_MHA = nn.LayerNorm(embed_dim//num_heads, eps=0.00001, bias=True, elementwise_affine=True, device=device)
        
        ##############
        # feed forward Network
        ##############
        
        self.FFN = FeedForwardNetwork(in_size=embed_dim, hidden_size=hidden_size, out_size=embed_dim, device=device)
        
        #normLayer
        self.normL_FFN = nn.LayerNorm(embed_dim, eps=0.00001, bias=True, elementwise_affine=True, device=device)
        
        ###############
        # output layer
        ###############
        
        self.outputLayer = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=output_features, bias=bias, device=device),
            nn.Softmax(dim=-1)
        )
        
        ###############
        # how is loss calculated
        ###############
        
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self,x,attn_mask=None,):
        """
        Method shows how data flows through the model
        
        This method also builds the nodes in the computation graph, 
        the computation graph takes care of tracking derivatives for backpropigation
        """
        embeddings = self.input_embeder(x)
        pos_encodings = self.pos_encoder(embeddings)
        residual = embeddings + pos_encodings
        
        attention_output, attention_weights = self.MHA.forward(residual, residual, residual, attn_mask=attn_mask,
                         average_attn_weights=False, need_weights=False)
        residual = self.normL_MHA(attention_output+residual)
        
        ffn_output = self.FFN(residual)
        residual = self.normL_FFN(ffn_output+residual)
        
        output = self.outputLayer(x)
        
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
        output = self.forward(input_tokens)
        loss = self.loss(output, labels)
        
        print(f'batch#:{batch_idx}, batch average loss:{torch.mean(loss)}')
                    
        return loss


# In[ ]:


class ASLDataset(Dataset):
    """
    This class helps us lazily load the data so memory doesn't blow up
    """
    def __init__(self, landmarkFile, oheFile, len, pdCacheSize):
        self.landmarkFile = landmarkFile
        self.oheFile = oheFile
        self.len = len
        
        #This is trying to implement a sort of cache using pandas, that way
        # we don't have to read from disc 
        self.pdCache:dict[str, pd.DataFrame] = {'landmarkDF':pd.DataFrame(), 'oheDF':pd.DataFrame()}
        self.pdCacheSize= pdCacheSize
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # if index is not in cache load a new block of cache starting with index
        if index not in self.pdCache['landmarkDF'].index:
            self.pdCache['landmarkDF'] = pd.read_csv(self.landmarkFile, skiprows=range(1,index+1), nrows=self.pdCacheSize)
            self.pdCache['oheDF'] = pd.read_csv(self.oheFile, skiprows=range(1,index+1), nrows=self.pdCacheSize)
            
        X = torch.from_numpy(self.pdCache['landmarkDF'].drop(columns=['Video file', 'Gloss']).to_numpy(dtype='float64'))
        y = torch.from_numpy(self.pdCache['oheDF'].drop(columns=['Video file', 'Gloss']).to_numpy(dtype='float64'))
        
        return X, y


# # Running Training code

# In[ ]:


longest_num_of_frames = 226
"""
Pose landmarks: 33 landmarks × (x,y,z,present) = 132 columns
Pose world landmarks: 33 landmarks × (x,y,z,present) = 132 columns
Left hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
Right hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
Overall presence indicators: 4 columns (poseLandmarks_present, poseWorldLandmarks_present, leftHandLandmarks_present, rightHandLandmarks_present)
Other columns: idx, Video file, Gloss
Total: 1+ 1 + 1 + 132 + 132 + 84 + 84 + 4 = 439 columns
Total input = 439 - 3 = 436 input columns
436 actual feature columns cause index, video file, label
((TotalInput - OverallPresenceIndicators)/4) + OverallPresenceIndicators = ((436-4)/4) + 4 = 112 embeddings
"""
n_embedings = 112 # divisible by 1, 2, 4, 7, 8, 14, 16, 28, 56, and 112
hiddenSize = n_embedings * 2
numberOfHeads = 4 # 112 % 4 == 0 is true
dropOutPercent = .1
device = "cuda" if torch.cuda.is_available() else "cpu"



model = Encoder(input_features=436, output_features=172, embed_dim=n_embedings, max_tokens=longest_num_of_frames*2,
                hidden_size=hiddenSize, num_heads=numberOfHeads, batch_first=True, dropOut=dropOutPercent,
                bias=False, device=device)


landmarkFile = '../data/Finished_Output.csv'
oheFile = '../data/onehot.csv'
numSamples = 20
pandasCacheSize = 10
dataset = ASLDataset(landmarkFile, oheFile,numSamples,pandasCacheSize)

batchSize = 5
shuffle = True
dataLoader = DataLoader(dataset=dataset, batch_size=batchSize,shuffle=shuffle)

trainer = L.Trainer(max_epochs=5)
trainer.fit(model, dataLoader)

