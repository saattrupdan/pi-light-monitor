'''Training script for the front-back model'''

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.utils.data as D
import numpy as np

from .model import FrontBackModel
from .dataset import FrontBackDataset
from ..utils import root_dir


def train(val_size: float = 0.2, batch_size: int = 2, random_seed: int = 4242):
    '''Train the front-back model'''
    # Initialise a random seed
    rng = np.random.default_rng(random_seed)

    # Load model, which uses a checkpoint if there is one available
    models_dir = root_dir / 'models'
    if len(list(models_dir.iterdir())) > 0:
        model_path = next(models_dir.glob('*.ckpt'))
        model = FrontBackModel.load_from_checkpoint(str(model_path))
    else:
        model = FrontBackModel()

    # Load dataset
    dataset = FrontBackDataset()

    # Split into train- and validation indices
    all_idxs = np.arange(len(dataset))
    val_idxs = rng.choice(all_idxs, size=int(len(dataset) * val_size))
    train_idxs = [idx for idx in all_idxs if idx not in val_idxs]

    # Create samplers for the train- and validation parts of the dataset
    train_sampler = D.SubsetRandomSampler(train_idxs)
    val_sampler = D.SubsetRandomSampler(val_idxs)

    # Build the dataloaders
    train_dl = D.DataLoader(dataset, batch_size=batch_size, num_workers=8,
                            sampler=train_sampler)
    val_dl = D.DataLoader(dataset, batch_size=batch_size, num_workers=8,
                          sampler=val_sampler)

    # Set up TensorBoard logging, model saving, and train the model
    tb_logger = TensorBoardLogger('tb_logs', name='front-back-model')
    model_checkpoint = ModelCheckpoint(dirpath='models', monitor='val_f1',
                                       mode='max')
    trainer = pl.Trainer(logger=tb_logger, callbacks=[model_checkpoint],
                         log_every_n_steps=1)
    trainer.fit(model, train_dl, val_dl)

    return model
