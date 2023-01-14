import argparse
import yaml
from dotmap import DotMap
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from models.emorec import EMOREC
from data.dataset import BaseDataset
from tqdm import tqdm
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))   


def cli_main():
    # ------------
    # args
    # ------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        action='store',
        dest='config',
        help='config.yaml',
        required=True)
    parser.add_argument(
        '--resume',
        action='store',
        dest='resume',
        help='resume training form ckpt',
        required=False)
    args = parser.parse_args()
    
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["emorec"])

    pl.seed_everything(1234)
    
    # ------------
    # init callbacks & logging
    # ------------
    checkpoint_callback = ModelCheckpoint(
        **config.checkpoint
    )
    logger = TensorBoardLogger(
        **config.logger
    )

    # ------------
    # data
    # ------------
    train_dataset = BaseDataset(
        path.join(config.train_behavior),
        path.join(config.train_news), 
        config)
    val_dataset = BaseDataset(
        path.join(config.val_behavior),
        path.join(config.train_news), 
        config) 
    train_loader = DataLoader(
        train_dataset,
        **config.train_dataloader)
    val_loader = DataLoader(
        val_dataset,
        **config.val_dataloader)
   
    # ------------
    # init model
    # ------------
    # load embedding pre-trained embedding weights
    embedding_weights=[]
    with open(config.embedding_weights, 'r') as file: 
        lines = file.readlines()
        for line in tqdm(lines):
            weights = [float(w) for w in line.split(" ")]
            embedding_weights.append(weights)
    pretrained_word_embedding = torch.from_numpy(
        np.array(embedding_weights, dtype=np.float32)
        )

    if config.name == "emorec":
        model = EMOREC(config, pretrained_word_embedding)
    # elif:
        # UPCOMING MODELS

    # ------------
    # training
    # ------------
    early_stop_callback = EarlyStopping(
       **config.early_stop
    )
    if args.resume is not None:
        model = model.load_from_checkpoint(
            args.resume, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
        trainer = Trainer(
            **config.trainer,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger,
            resume_from_checkpoint=args.resume
        )
    else:
        trainer = Trainer(
            **config.trainer,
            callbacks=[early_stop_callback, checkpoint_callback],
            logger=logger
        )
    trainer.fit(
        model=model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader)

if __name__ == '__main__':
    cli_main()