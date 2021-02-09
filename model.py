import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F

from cnn import CNN

from argparse import ArgumentParser

from torch.optim import Adam


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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # transform token ids into embeddings:
        # (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.dropout_1(self.embed(x))

        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        embedded = embedded.permute(1, 0, 2)

        # pass embedded tokens (batch, max_len, embed_dim) through bi-RNN
        # take final state (batch, 2*rnn_dim)
        h_concat, hidden = self.rnn(embedded)
        # apply dropout to output
        h_final= self.dropout_2(h_concat)

        # calculate probabilities for z
        probs = self.__z_forward(h_final)

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

        self.hidden = nn.Linear(2*args.rnn_dim, num_classes)


    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        
        embedded = self.embed(x) * mask.T.unsqueeze(-1)
        # permute (batch, seq_len, embed_dim) -> (seq_len, batch, embed_dim)
        embedded = embedded.permute(1, 0, 2)
        # pass embedded tokens (batch, max_len, embed_dim) through bi-RNN
        # take final state (batch, 2*rnn_dim)
        h, hidden = self.rnn(embedded)
        # apply dropout to output
        h= self.dropout_2(h)

        h_concat = torch.cat((h[:,0, 128:], h[:,-1, :128]), dim=1)

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

    def forward(self, x: torch.Tensor):
        mask = self.gen(x)
        logits = self.enc(x, mask)
        
        return logits

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.args.learning_rate)

    def training_step(self, batch, batch_idx):
        x,y = batch
        logits = self(x)

        loss = F.cross_entropy(logits, y) * self.gen.logpz
        loss += self.continuity_lambda * self.gen.zdiff
        loss += self.selection_lambda * self.gen.zsum

        # multiply loss by log prob

        return loss

        