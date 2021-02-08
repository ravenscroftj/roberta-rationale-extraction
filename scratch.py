# %%
import model
import torch

emb = torch.empty((5000,128)).uniform_()
emb[0] = torch.empty((128,)).zero_()
# %%
gen = model.GeneratorModel(emb, rnn_layers=2, rnn_dim=128, dropout=0.5).cuda()
# %%

x = torch.empty((10, 256)).random_(0,5000).type(torch.LongTensor).cuda()

# %%
mask = gen.forward(x).cuda()
# %%
enc = model.Encoder(emb, 2).cuda()
# %%
h = enc(x, mask)

# %%
import torch.nn.functional as F
y = torch.argmax(torch.rand_like(h), dim=1)

# %%
m = model.RationaleSystem(gen, enc).cuda()

m.training_step((x,y), None)

# %%

# %%
