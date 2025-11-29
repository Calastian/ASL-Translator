# %% [markdown]
# """
# Pose landmarks: 33 landmarks × (x,y,z,present) = 132 columns
# Included pose landmarks: 23 landmarks - 10 landmarks = 23 landmarks x (x,y,z,present) = 92 columns
# Pose world landmarks: 33 landmarks × (x,y,z,present) = 132 columns
# Included pose world landmarks: 23 landmarks - 10 landmarks = 23 landmarks x (x,y,z,present) = 92 columns
# Included Left hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
# Included Right hand landmarks: 21 landmarks × (x,y,z,present) = 84 columns
# Overall presence indicators: 4 columns (poseLandmarks_present, poseWorldLandmarks_present, leftHandLandmarks_present, rightHandLandmarks_present)
# excluded columns: idx, Video file, Gloss
# included columns: padding
# Total: 92 + 92 + 84 + 84 + 4 + 1 + 1 + 1 + 1 = 360 columns
# Total input = 360 - 3 = 357 input columns
# 357 actual feature columns cause index, video file, label
# ((TotalInput - OverallPresenceIndicators)/5) + OverallPresenceIndicators = ((357-5)/4) + 5 = 93 embeddings
# """

#%%
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

N_INPUTS = 357
N_OUTPUTS = 172
MAX_TOKENS = 266
N_EMBEDDINGS = 93

BIAS = False

HIDDEN_SIZE = N_EMBEDDINGS * 2 # this changes size of hidden layers in the feed forward neural network
N_HEADS = 3 #number of heads needs to be a factor of N_EMBEDDINGS

EPS = .0001 #for normalization layers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#for learning
EPOCS = 10000
DROP_OUT = 0.1
LR = .0001

BATCH_SIZE = 128
N_WORKERS = 2

DATA_DIRECTORY = './training_data/'
DATA_PREFIX = '' #use 'small_' for the small data 

TRAIN_X_FILE = DATA_DIRECTORY + DATA_PREFIX + "padd_training.csv"
TRAIN_Y_FILE = DATA_DIRECTORY + DATA_PREFIX + "training_encoding.csv"
TRAIN_N_SAMPLES = 2444

VAL_X_FILE = DATA_DIRECTORY + DATA_PREFIX + "padd_val.csv" 
VAL_Y_FILE = DATA_DIRECTORY + DATA_PREFIX + "val_encoding.csv"
VAL_N_SAMPLES = 624

LOGGER = TensorBoardLogger('tb_log', 'model_V0')

CHECKPOINTS = [ModelCheckpoint(
        dirpath="../models/",
        filename="ASL_Model_",
        save_top_k =20,
        monitor="val_loss",
        mode="min"
)]

ACCELERATOR = 'gpu'
DEVICES = 1 #set to negative one to use all available devices

#Over fit batch to make check pipeline and bottle necks
OVERFIT_BATCH = False
OVERFIT_EPOCS = 100
OVERFIT_LOGGER = TensorBoardLogger('OverfitLogs', 'BatchOverFitModel')
OVERFIT_CHECKPOINTS = [ModelCheckpoint(
        dirpath="../models/",
        filename="ASL_Model_",
        save_top_k =20,
        monitor="val_loss",
        mode="min"
)]
OVERFIT_N_BATCHES = 1

# inference
MODEL_FILE = '../models/ASL_Model_.ckpt'
INPUT_FILE = DATA_DIRECTORY + DATA_PREFIX + "padd_val.csv" # can be changed to whatever input file, just make sure data needed to be droped is in the COLS_TO_DROP constant
COLS_TO_DROP = ['Video file', 'Gloss']
N_PREDICTIONS = 10
