from torch import nn, softmax

from transformers import RobertaForSequenceClassification

class RoBERTaSentimentClassifier(nn.Module):

    def __init__(self, device, base_model='roberta-base'):

        super(RoBERTaSentimentClassifier, self).__init__()

        self.encoder = RobertaForSequenceClassification.from_pretrained(base_model).to(device)
        self.device = self.encoder.device


    def forward(self, text, label):

        loss, text_fea = self.encoder(text, labels=label)[:2]

        return loss, text_fea