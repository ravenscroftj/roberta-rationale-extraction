import os
import torch
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader

from model import GeneratorModel, Encoder, RationaleSystem

from argparse import ArgumentParser


from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB
from torchtext.vocab import FastText
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import IMDBDataModule

def general_args():
    ap = ArgumentParser()
    
    ap.add_argument("--no-generator", action="store_true", default=False)

    ap.add_argument('--learning-rate', type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=32)

    ap.add_argument("--rnn-type", type=str, choices=['gru','cnn'])

    ap.add_argument('--rnn-layers', type=int, default=2)
    ap.add_argument('--rnn-dim', type=int, default=128)
    ap.add_argument('--dropout', type=float, default=0.2)

    ap.add_argument("--cnn-filters", type=str, default="3,4,5")

    ap.add_argument("--patience", default=3, type=int)
    

    ap.add_argument("--neptune-project", type=str, default=None)
    ap.add_argument("--neptune-key", type=str, default=None)

    return ap

def main():
    parser = general_args()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = RationaleSystem.add_model_specific_args(parser)
    

    args = parser.parse_args()

    neptune_logger = None

    if args.neptune_project is not None:
        from pytorch_lightning.loggers.neptune import NeptuneLogger

        neptune_logger = NeptuneLogger(
            api_key=args.neptune_key,
            project_name=args.neptune_project,
            params=vars(args)
        )

    

    data = IMDBDataModule(args.batch_size)
    data.prepare_data()

    # if args.no_generator:
    #     gen = None
    # else:
    #     gen = GeneratorModel(args, 
    #     embeddings=data.text_field.vocab.vectors, 
    #     padding_idx=data.text_field.vocab.stoi['<pad>'])


    # enc = Encoder(args,
    # embeddings=data.text_field.vocab.vectors, 
    # num_classes=len(data.label_field.vocab), 
    # padding_idx=data.text_field.vocab.stoi['<pad>'])

    model = RationaleSystem(args, embeddings=data.text_field.vocab.vectors, 
    num_classes=len(data.label_field.vocab), 
    padding_idx=data.text_field.vocab.stoi['<pad>'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=3,
        save_weights_only=True,
        verbose=True,
        monitor='val_acc',
        mode='max',
        prefix=''
    )

    earlystop_callback = EarlyStopping(monitor='val_acc', patience=args.patience)


    trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback, earlystop_callback], logger=neptune_logger)

    trainer.fit(model, data)



if __name__ == "__main__":
    main()