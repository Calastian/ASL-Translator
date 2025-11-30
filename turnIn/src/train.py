# %%

#third party
import torch

from torch.utils.data import DataLoader

import lightning as L

#our imports
from architecture import Encoder
from dataSet import PreloadedASLDataset
import config

# %% [markdown]
# # Running Training code

# %%
if __name__ == "__main__":    
    model = Encoder(input_features=config.N_INPUTS, output_features=config.N_OUTPUTS,
                    embed_dim=config.N_EMBEDDINGS, max_tokens=config.MAX_TOKENS,
                    hidden_size=config.HIDDEN_SIZE, num_heads=config.N_HEADS,
                    bias=config.BIAS, eps=config.EPS, device=config.DEVICE,
                    dropOut=config.DROP_OUT, lr=config.LR)
     
    dataset = PreloadedASLDataset(config.TRAIN_X_FILE, config.TRAIN_Y_FILE,
                                  length=config.TRAIN_N_SAMPLES, device=config.DEVICE)    
    dataLoader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                            num_workers=config.N_WORKERS, persistent_workers=True)
    
    val_dataset = PreloadedASLDataset(config.VAL_X_FILE, config.VAL_Y_FILE,
                                  length=config.VAL_N_SAMPLES, device=config.DEVICE)    
    val_dataLoader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            num_workers=config.N_WORKERS, persistent_workers=True)
    
    if config.OVERFIT_BATCH:
        overfitTrainer = L.Trainer(logger=config.OVERFIT_LOGGER,max_epochs=config.OVERFIT_EPOCS,
                                   accelerator=config.ACCELERATOR, devices=config.DEVICES,
                                   callbacks=config.OVERFIT_CHECKPOINTS,
                                   overfit_batches=config.OVERFIT_N_BATCHES)
        overfitTrainer.fit(model, dataLoader, val_dataLoader)
    else:
        trainer = L.Trainer(logger=config.LOGGER, max_epochs=config.EPOCS,
                            accelerator=config.ACCELERATOR, devices=config.DEVICES,
                            callbacks=config.CHECKPOINTS)
        trainer.fit(model, dataLoader, val_dataLoader)
