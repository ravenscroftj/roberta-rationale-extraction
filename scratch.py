# %%
import torch
from model import GeneratorModel, Encoder, RationaleSystem


# %%
from data import IMDBDataModule
data = IMDBDataModule(32)
data.setup()

# %%
from train import general_args

ap = general_args()
args = ap.parse_args("")
# %%


gen = GeneratorModel(args, 
embeddings=data.text_field.vocab.vectors, 
padding_idx=data.text_field.vocab.stoi['<pad>'])

enc = Encoder(args,
embeddings=data.text_field.vocab.vectors, 
num_classes=len(data.label_field.vocab), 
padding_idx=data.text_field.vocab.stoi['<pad>'])

model = RationaleSystem(args, gen, enc)
# %%
batch = next(iter(data.train_dataloader()))
loss = model.training_step(batch, 0)
# %%
loss.backward()