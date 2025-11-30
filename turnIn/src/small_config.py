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

#%%
#model characteristics
N_INPUTS = 357
N_OUTPUTS = 172
MAX_TOKENS = 266
N_EMBEDDINGS = 93

BIAS = False

HIDDEN_SIZE = N_EMBEDDINGS * 2 # this changes size of hidden layers in the feed forward neural network
N_HEADS = 3 #number of heads needs to be a factor of N_EMBEDDINGS

EPS = .0001 #for normalization layers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%
#for learning
EPOCS = 100
DROP_OUT = 0.1
LR = .0001

BATCH_SIZE = 20
N_WORKERS = 2

PREFIX_DIRECTORY= 'small/' #choose which 

DATA_DIRECTORY = './training_data/'

TRAIN_X_FILE = DATA_DIRECTORY + PREFIX_DIRECTORY + "padd_train.csv"
TRAIN_Y_FILE = DATA_DIRECTORY + PREFIX_DIRECTORY + "train_encoding.csv"
TRAIN_N_SAMPLES = 80

VAL_X_FILE = DATA_DIRECTORY + PREFIX_DIRECTORY + "padd_val.csv" 
VAL_Y_FILE = DATA_DIRECTORY + PREFIX_DIRECTORY + "val_encoding.csv"
VAL_N_SAMPLES = 20

MODEL_DIRECTORY = './models/' 
MODEL_PREFIX_DIRECTORY = '' #change to 'small/ for small models stuff

MODEL_NAME = 'ASL_Model_'

LOGGER = TensorBoardLogger(MODEL_DIRECTORY + PREFIX_DIRECTORY + 'tb_log', MODEL_NAME)

CHECKPOINTS = [ModelCheckpoint(
        dirpath=MODEL_DIRECTORY+PREFIX_DIRECTORY,
        filename=MODEL_NAME,
        save_top_k =5,
        monitor="val_loss",
        mode="min"
)]

# ACCELERATOR = 'gpu'
ACCELERATOR = 'auto'
DEVICES = 1 #set to negative one to use all available devices

#%%
#Over fit batch to make check pipeline and bottle necks
#Need to run training cell first
OVERFIT_BATCH = False
OVERFIT_EPOCS = 100
OVERFIT_LOGGER = TensorBoardLogger(MODEL_DIRECTORY+PREFIX_DIRECTORY+'OverfitLogs', MODEL_NAME)
OVERFIT_CHECKPOINTS = [ModelCheckpoint(
        dirpath=MODEL_DIRECTORY+PREFIX_DIRECTORY,
        filename=MODEL_NAME,
        save_top_k =20,
        monitor="val_loss",
        mode="min"
)]
OVERFIT_N_BATCHES = 1

#%%
# inference
MODEL_FILE = './models/small/ASL_Model_.ckpt'

INPUT_FILE = DATA_DIRECTORY + PREFIX_DIRECTORY + "padd_val.csv" # can be changed to whatever input file, just make sure data needed to be droped is in the COLS_TO_DROP constant
COLS_TO_DROP = ['Video file', 'Gloss']
N_PREDICTIONS = 10

#key file
KEY_FILE = '../docs/Key_ASL.csv'