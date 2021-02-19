# %%
import torch
from model import GeneratorModel, Encoder, RationaleSystem


# %%
from data import IMDBDataModule
data = IMDBDataModule(32)
data.prepare_data()

# %%
from train import general_args

ap = general_args()
args = ap.parse_args(["--no-generator"])
# %%


# gen = GeneratorModel(args, 
# embeddings=data.text_field.vocab.vectors, 
# padding_idx=data.text_field.vocab.stoi['<pad>'])

# enc = Encoder(args,
# embeddings=data.text_field.vocab.vectors, 
# num_classes=len(data.label_field.vocab), 
# padding_idx=data.text_field.vocab.stoi['<pad>'])

model = RationaleSystem(args, 
embeddings=data.text_field.vocab.vectors,
padding_idx=data.text_field.vocab.stoi['<pad>'],
num_classes=len(data.label_field.vocab))
# %%
batch = next(iter(data.train_dataloader()))
# %%
model.training_step(batch, 0)
# %%
%load_ext tensorboard
# %%
%tensorboard --logdir lightning_logs/
# %%
weights = torch.load("epoch=24-step=13674.ckpt")
# %%
model.load_state_dict(weights['state_dict'])
# %%
model.eval()

#model.predict()
# %%
tok = data.text_field.tokenize("I really enjoyed this film it had really good acting")
x, lens = data.text_field.process([tok])
#%%
lens
# %%
model(x,lens)
# %%
len("the movie was crap and I didn't care for it")
# %%
x
# %%
tensor = torch.rand((200, 32)).permute(1,0)

# %%
tensor.shape
# %%
lens = torch.randint(low=0,high=199, size=(32,))
# %%
lens
# %%
lens.shape
# %%
lens.unsqueeze(0)
# %%
lens
# %%
mask = (torch.arange(200)[None, :] < lens[:, None]).to(torch.int)
# %%
mask.shape
# %%
mask
# %%
mask[0][143]
# %%
