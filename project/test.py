import argparse
import yaml
from dotmap import DotMap
from os import path
import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from models.emorec import EMOREC
from data.dataset import BaseDataset
from tqdm import tqdm


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
        '--ckpt',
        action='store',
        dest='ckpt',
        help='checkpoint to load',
        required=True)
    args = parser.parse_args()
    
    with open(args.config, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
        config = DotMap(config)

    assert(config.name in ["emorec"])

    pl.seed_everything(1234)
    
    # ------------
    # logging
    # ------------
    logger = TensorBoardLogger(
        **config.logger
    )
    # logger = CSVLogger(
    #     **config.logger
    # )

    # ------------
    # data
    # ------------

    test_dataset = BaseDataset(
        path.join(config.test_behavior),
        path.join(config.test_news), 
        config)
    test_loader = DataLoader(
        test_dataset,
        **config.test_dataloader)
   
    #print(len(dataset), len(train_dataset), len(val_dataset))
    # ------------
    # init model
    # ------------
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
        model = EMOREC.load_from_checkpoint(
            args.ckpt, 
            config=config, 
            pretrained_word_embedding=pretrained_word_embedding)
    # elif:
        # UPCOMING MODELS

    # ------------
    # Test
    # ------------
    trainer = Trainer(
        **config.trainer,
        logger=logger
    )

    trainer.test(
        model=model, 
        dataloaders=test_loader
    )
    # trainer.test()

if __name__ == '__main__':
    cli_main()