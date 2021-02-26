import argparse
import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

from cnn import CNN

from argparse import ArgumentParser

from typing import Optional

from torch.optim import Adam

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GeneratorModel(pl.LightningModule):


    def __init__(self, args, embeddings: torch.Tensor, padding_idx:int=0):
        super(GeneratorModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(
            embeddings, padding_idx=padding_idx, freeze=True)

        #self.dropout_1 = nn.Dropout(args.dropout)

        self.rnn_dim = args.rnn_dim

        self.rnn_type = args.rnn_type

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.embed.embedding_dim,
                            hidden_size=args.rnn_dim,
                            num_layers=args.rnn_layers,
                            bidirectional=True)

            self.z_prob_1 = nn.Linear(2*args.rnn_dim, 1)

        elif self.rnn_type == "cnn":
            self.rnn = CNN(embedding_dim=self.embed.embedding_dim, 
                num_layers=args.rnn_layers, 
                filters=[int(x) for x in args.cnn_filters.split(",")],
                filter_num=args.rnn_dim,)

            layercount = len(args.cnn_filters.split(","))
            self.fc = nn.Linear(layercount*args.rnn_dim, args.rnn_dim)
            self.z_prob_1 = nn.Linear(args.rnn_dim, 1)




    def __z_forward(self, x: torch.Tensor, lens: torch.Tensor):
        
        seqlen = x.shape[0]

        z = torch.sigmoid(self.z_prob_1(x).squeeze(-1))

        mask = (torch.arange(seqlen, device=x.device)[None, :] < lens.to(device=x.device).unsqueeze(1)[:, None]).squeeze(1).to(x.device, dtype=torch.int)

        # mask zeros longer 

        return z * mask.T


    def _generate_probs(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        """Shared function used by inference and training to generate probs for rationale"""
        # transform token ids into embeddings:
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embed(x)

        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        #embedded = embedded.permute(1, 0, 2)

        if self.rnn_type == "gru":

            packed = pack_padded_sequence(embedded, lens, batch_first=False)

            h, hidden = self.rnn(packed)
            # apply dropout to output

            unpacked, _ = pad_packed_sequence(h, batch_first=False, total_length=x.shape[0])

        elif self.rnn_type == "cnn":

            embedded = embedded.permute(0, 2, 1)
            unpacked = self.rnn(embedded)
            unpacked = unpacked.permute(0, 2, 1)
            unpacked = F.relu(self.fc(unpacked))


        # calculate probabilities for z
        probs = self.__z_forward(unpacked, lens)

        return probs

    def training_step(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
     
        probs = self._generate_probs(x, lens)
        
        # now we sample rationales for this batch
        mask = torch.bernoulli(probs).detach()
        
        # calculate prob (probability - 1 or 0 for each token)
        # using this in our loss term allows us to backprop and 
        # change the weights in z_prob_1
        self.logpz = F.binary_cross_entropy(probs, mask, reduce=False)

        self.probs = probs

        # term 1 in rationale regularizer - penalise long summaries
        self.zsum = mask.sum(dim=0)

        l_padded_mask =  torch.cat( [mask[0,:].unsqueeze(0), mask] , dim=0)
        r_padded_mask =  torch.cat( [mask, mask[-1,:].unsqueeze(0)] , dim=0)


        # term 2 in rationale regularizer - penalise incoherent summaries
        self.zdiff = torch.sum(torch.abs(l_padded_mask - r_padded_mask), dim=0)

        return mask


    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:

        # calculate probabilities for z
        probs = self._generate_probs(x, lens)

        # in production we just take 'most likely' mask
        # we don't want randomness in our predictions any more
        #mask = (probs>0.5).float()
        mask = torch.bernoulli(probs).detach()
        
        return mask


class Encoder(pl.LightningModule):

    def __init__(self, args, embeddings, num_classes: int, padding_idx=0):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(
            embeddings, padding_idx=padding_idx, freeze=True)

        self.rnn_dim = args.rnn_dim

        self.rnn_type = args.rnn_type

        if self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.embed.embedding_dim,
                            hidden_size=args.rnn_dim,
                            num_layers=args.rnn_layers,
                            bidirectional=True)

            self.hidden = nn.Linear(2*args.rnn_dim, num_classes)

        elif self.rnn_type == "cnn":

            self.rnn = CNN(embedding_dim=self.embed.embedding_dim, 
                num_layers=args.rnn_layers, 
                filters=[int(x) for x in args.cnn_filters.split(",")],
                filter_num=args.rnn_dim, max_pool_over_time=True)

            layercount = len(args.cnn_filters.split(","))
            
            self.fc = nn.Linear(layercount*args.rnn_dim, args.rnn_dim)
            self.hidden = nn.Linear(args.rnn_dim, num_classes)

        self.dropout = nn.Dropout(args.dropout)

    def training_step(self, x: torch.Tensor, lens, mask: Optional[torch.Tensor] = None):
        return self.forward(x, lens, mask)


    def forward(self, x: torch.Tensor, lens, mask: Optional[torch.Tensor] = None):
        
        embedded = self.embed(x).permute(1,0,2)

        if mask is not None:
            embedded *= mask.T.unsqueeze(-1)

        embedded = self.dropout(embedded)
            

        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        #embedded = embedded.permute(1, 0, 2)
        # pass embedded tokens (batch, max_len, embed_dim) through bi-RNN
        # take final state (batch, 2*rnn_dim)

        if self.rnn_type == "gru":

            packed = pack_padded_sequence(embedded, lens, batch_first=True)

            h, hidden = self.rnn(packed)
            # apply dropout to output

            unpacked, _ = pad_packed_sequence(h, batch_first=True, total_length=x.shape[0])

            hidden = unpacked[:,-1, :]

        elif self.rnn_type == "cnn":

            embedded = embedded.permute(0, 2, 1)
            unpacked = self.rnn(embedded)
            #unpacked = unpacked.permute(2, 0, 1)

            hidden = F.relu(self.fc(unpacked))

        hidden = self.dropout(hidden)
        logit = self.hidden(hidden)

        return F.softmax(logit, dim=1)


class RationaleSystem(pl.LightningModule):

    @classmethod
    def add_model_specific_args(cls, ap: argparse.ArgumentParser):
        """Add model specific args for rationale model"""
        ap.add_argument("--continuity-lambda", type=float, default=0.1)
        ap.add_argument("--selection-lambda", type=float, default=0.1)

        return ap

    def __init__(self, args, embeddings, num_classes, padding_idx):
        super().__init__()
        #self.save_hyperparameters()
        self.args = args
        self.gen = GeneratorModel(args, embeddings, padding_idx)
        self.enc = Encoder(args, embeddings, num_classes, padding_idx)
        self.celoss = nn.CrossEntropyLoss()
        self.continuity_lambda = args.continuity_lambda
        self.selection_lambda = args.selection_lambda

        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()

        self.epoch_train_loss = 0
        self.epoch_val_loss = 0


    def forward(self, x: torch.Tensor, lens: torch.Tensor):

        lens = lens.to(torch.int64).cpu()

        mask = None
        if self.gen is not None:
            mask = self.gen(x, lens)

        logits = self.enc(x, lens, mask)
        
        return logits, mask

    def __calculate_loss(self, logits, y):
        """Calculate loss for predicted logits"""
        
        enc_loss = F.cross_entropy(logits, y, reduce=False)

        gen_loss = 0
        if self.gen is not None:

            lasso_loss = self.gen.probs.mean(axis=0)

            gen_loss = ( 
                enc_loss.detach()
                + lasso_loss
                + (self.continuity_lambda * self.gen.zdiff)
                + (self.selection_lambda * self.gen.zsum)
            ) * self.gen.logpz.mean(0)

        return enc_loss.mean() + gen_loss.mean()
 

    def training_step(self, batch, batch_idx):
        (x,lens), y = batch

        lens = lens.to(torch.int64).cpu()

        mask = None
        if self.gen is not None:
            mask = self.gen.training_step(x, lens)

        logits = self.enc.training_step(x, lens, mask)

        loss = self.__calculate_loss(logits, y)

        self.log('train_acc_step', self.accuracy(logits, y))
        self.epoch_train_loss += loss

        return loss

    def training_epoch_end(self, outputs) -> None:

        self.log('train_acc', self.accuracy.compute(), prog_bar=True)
        self.log('epoch_loss', self.epoch_train_loss, prog_bar=True)
        self.epoch_train_loss = 0
        self.accuracy.reset()
        

    def validation_step(self, batch, batch_idx):
        (x,lens), y = batch
        logits, _ = self(x, lens)

        #loss = self.__calculate_loss(logits, y)

        self.log('val_acc_step', self.val_accuracy(logits, y))
        #self.epoch_val_loss += loss
        #return loss


    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)
        #self.log('val_loss', self.epoch_val_loss, prog_bar=True)
        #self.epoch_val_loss = 0
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.learning_rate)