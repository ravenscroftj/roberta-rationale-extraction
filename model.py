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


    def __init__(self, args, embeddings: torch.Tensor, padding_idx=0):
        super(GeneratorModel, self).__init__()

        self.embed = nn.Embedding.from_pretrained(
            embeddings, padding_idx=padding_idx, freeze=True)

        self.dropout_1 = nn.Dropout(args.dropout)

        self.rnn_dim = args.rnn_dim

        self.rnn = nn.GRU(input_size=self.embed.embedding_dim,
                          hidden_size=args.rnn_dim,
                          num_layers=args.rnn_layers,
                          bidirectional=True)

        #self.rnn = CNN(embedding_dim=self.embed.embedding_dim, num_layers=1)

        self.dropout_2 = nn.Dropout(args.dropout)

        # linear dense layer applied to each token during sampling
        self.z_prob_1 = nn.Linear(2*args.rnn_dim, 1)
        # linear dense layer applied to hidden state during sampling



    def __z_forward(self, x: torch.Tensor):
        
        seqlen, batchlen, _ = x.shape

        z = torch.zeros((seqlen, batchlen)).to(x.device)

        for i in range(seqlen):
           z[i] = torch.sigmoid(self.z_prob_1(x[i, :, :])).squeeze()

        return z


    def forward(self, x: torch.Tensor, lens: torch.Tensor) -> torch.Tensor:
        # transform token ids into embeddings:
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.dropout_1(self.embed(x))

        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        #embedded = embedded.permute(1, 0, 2)

        packed = pack_padded_sequence(embedded, lens, batch_first=False)

        h, hidden = self.rnn(packed)
        # apply dropout to output

        unpacked, _ = pad_packed_sequence(h, batch_first=False, total_length=x.shape[0])

        unpacked = self.dropout_2(unpacked)

        #h_concat = unpacked[:,-1, :]

        # calculate probabilities for z
        probs = self.__z_forward(unpacked)

        # now we sample rationales for this batch
        mask = torch.bernoulli(probs).detach()
        
        # calculate prob (probability - 1 or 0 for each token)
        # using this in our loss term allows us to backprop and 
        # change the weights in z_prob_1
        self.logpz = F.binary_cross_entropy(probs, mask)

        # term 1 in rationale regularizer - penalise long summaries
        self.zsum = torch.mean(mask.sum(dim=1))

        l_padded_mask =  torch.cat( [mask[:,0].unsqueeze(1), mask] , dim=1)
        r_padded_mask =  torch.cat( [mask, mask[:,-1].unsqueeze(1)] , dim=1)


        # term 2 in rationale regularizer - penalise incoherent summaries
        self.zdiff = torch.mean( torch.sum( torch.abs( l_padded_mask - r_padded_mask ) , dim=1) )

        return mask


class Encoder(pl.LightningModule):

    def __init__(self, args, embeddings, num_classes: int, padding_idx=0):
        super(Encoder, self).__init__()

        self.embed = nn.Embedding.from_pretrained(
            embeddings, padding_idx=padding_idx, freeze=True)

        self.rnn = nn.GRU(input_size=self.embed.embedding_dim,
                          hidden_size=args.rnn_dim,
                          num_layers=args.rnn_layers,
                          bidirectional=True)

        self.dropout_2 = nn.Dropout(args.dropout)

        self.rnn_dim = args.rnn_dim

        self.hidden = nn.Linear(2*args.rnn_dim, num_classes)


    def forward(self, x: torch.Tensor, lens, mask: Optional[torch.Tensor] = None):
        
        embedded = self.embed(x).permute(1,0,2)

        if mask is not None:
            embedded *= mask.T.unsqueeze(-1)
            

        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        #embedded = embedded.permute(1, 0, 2)
        # pass embedded tokens (batch, max_len, embed_dim) through bi-RNN
        # take final state (batch, 2*rnn_dim)
        packed = pack_padded_sequence(embedded, lens, batch_first=True)

        h, hidden = self.rnn(packed)
        # apply dropout to output

        unpacked, _ = pad_packed_sequence(h, batch_first=True)

        unpacked = self.dropout_2(unpacked)

        h_concat = unpacked[:,-1, :]

        logit = self.hidden(h_concat)

        return F.softmax(logit, dim=1)


class RationaleSystem(pl.LightningModule):

    def __init__(self, args, gen: nn.Module, enc: nn.Module):
        super().__init__()
        self.args = args
        self.gen = gen
        self.enc = enc
        self.celoss = nn.CrossEntropyLoss()
        self.continuity_lambda = .01
        self.selection_lambda = .01

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
        
        return logits

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.learning_rate)
 
    def __forward_step(self, batch):
        (x,lens),y = batch
        logits = self(x,lens)

        loss = F.cross_entropy(logits, y)

        if self.gen is not None:

            loss * self.gen.logpz
            loss += self.continuity_lambda * self.gen.zdiff
            loss += self.selection_lambda * self.gen.zsum

        return logits, loss

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self.__forward_step(batch)
        self.log('train_acc_step', self.accuracy(logits, y))
        self.epoch_train_loss += loss
        return loss

    def validation_step(self, batch, batch_idx):
        x,y = batch
        logits, loss = self.__forward_step(batch)
        self.log('val_acc_step', self.val_accuracy(logits, y))
        self.epoch_val_loss += loss
        return loss


    def training_epoch_end(self, outputs) -> None:

        self.log('train_acc', self.accuracy.compute(), prog_bar=True)
        self.log('epoch_loss', self.epoch_train_loss, prog_bar=True)
        self.epoch_train_loss = 0
        self.accuracy.reset()
        


    def validation_epoch_end(self, outputs) -> None:
        self.log('val_acc', self.val_accuracy.compute(), prog_bar=True)
        self.epoch_val_loss = 0
        self.val_accuracy.reset()