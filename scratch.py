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
ap = RationaleSystem.add_model_specific_args(ap)
args = ap.parse_args(["--no-generator", "--rnn-layers=1", "--rnn-type=cnn"])
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
loss = model.training_step(batch, 0)
#%%
loss.backward()
#%%
model.validation_step(batch,0)
# %%
%load_ext tensorboard
# %%
%tensorboard --logdir lightning_logs/
# %%
weights = torch.load("/home/james/workspace/movie-rationale/epoch=8-step=4922.ckpt")
# %%
model.load_state_dict(weights['state_dict'])
# %%
model.eval()

# %%

# %%
review = """‘Crock of Gold: A Few Rounds With Shane MacGowan’

Though he’s hardly a household name, the Irish singer-songwriter is responsible for one of the best-known songs of our time: the controversial modern Christmas standard “Fairytale of New York,” which he co-wrote with Jem Finer of his band the Pogues, and which he sang alongside the great vocalist Kirsty MacColl. Fans of punk, folk and traditional Irish music know too that MacGowan and the Pogues were responsible for some of the most thrilling and beautiful music of the 1980s, marrying the imagery and insight of great literature with the frenzied stomp of rock ’n’ roll.

Those same fans are probably also aware that MacGowan has a reputation as … well, a bit of a tippler. Director Julien Temple makes a joke of that with the full title of his documentary. MacGowan has reportedly eased off considerably from the days when his prodigious drinking got him kicked out of the Pogues — who by the ‘90s found it harder to work around their frontman’s frequent onstage descent into incoherence. Nevertheless, he appears in the film decidedly impaired by a lifetime of ill health and hedonism.

“Crock of Gold” isn’t intended as a lament for an artist derailed by his worst impulses, though. Instead, it’s a celebration of what MacGowan accomplished at his peak, as well as an explanation of the experiences that informed his music. Temple has access to a wealth of old footage, including a lot from the days when MacGowan was a more energetic performer — and a more articulate interview subject. He combines this with animation and clever clips from old movies, similar to what he did with his acclaimed 2000 Sex Pistols documentary “The Filth and the Fury.”

Running more than two hours, “Crock of Gold” will probably appeal most to those who already love MacGowan’s songs, especially since Temple doesn’t really get to the Pogues era until the second half. Still, for those devotees — and for anyone who one day ventures beyond “Fairytale” — this movie is a must. It’s a rousing and illuminating tribute to a brilliant musician who burned out quickly, but burned so brightly."""

#%%
review = "The film was good and I did enjoy it"

#%%
review = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in."


tok = data.text_field.tokenize(review)
# %%
len(tok)
# %%
x, lens = data.text_field.process([tok])
kls, mask = model(x,lens)
kls

# %%
[[t if x == 1 else "" for (t,x) in zip(tok, mask) ]]
# %%
# %%
sum(mask)
# %%
simplebatch = [(x,torch.tensor([len(tok)])), torch.tensor([0]) ]
# %%
model.training_step(simplebatch,0)
# %%
