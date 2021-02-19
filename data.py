import random
import pytorch_lightning as pl

from torch.utils.data.dataloader import DataLoader

from torchtext.data import Field, BucketIterator
from torchtext.data.field import LabelField
from torchtext.vocab import FastText, GloVe
from torchtext.datasets import IMDB

class IMDBDataModule(pl.LightningDataModule):

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size=batch_size

    def prepare_data(self):
        self.text_field = Field(sequential=True, fix_length=200, include_lengths=True)
        self.label_field = LabelField()

        train_val, test = IMDB.splits(self.text_field, self.label_field)
        random.seed(42)
        train, val = train_val.split(random_state=random.getstate())

        self.text_field.build_vocab(train, vectors=GloVe())#vectors=FastText('simple'))
        self.label_field.build_vocab(train)

        

        self.train_iter, self.test_iter, self.val_iter = BucketIterator.splits(
            (train, test, val), 
            batch_size=self.batch_size
        )

        self.train_iter.sort_within_batch = True
        self.val_iter.sort_within_batch = True


    def train_dataloader(self) -> DataLoader:
        return self.train_iter

    def test_dataloader(self) -> DataLoader:
        return self.test_iter

    def val_dataloader(self)  -> DataLoader:
        return self.val_iter