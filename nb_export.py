import argparse
# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from pkl_io import save_pkl, load_pkl
from sklearn.manifold import TSNE
import scipy.io as sio
import pandas as pd
import numpy as np
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(1)


print("here")
# %%
DIM_SIZE = 200
NUM_NEG_SAMPLES = 1


# %%
lex = sio.loadmat('data/gold_standard.mat')['gold_standard']['map']
world = sio.loadmat('data/world.mat')['world']
corpus = sio.loadmat('data/corpus.mat')['corpus']

# %% [markdown]
# # World

# %%
def get_world():
    
    world = sio.loadmat('data/world.mat')['world']
    world = world[0][0]
    
    word_idx = world[0][0] - 1
    words = list(itertools.chain.from_iterable(map(list, world[1][0])))  
    word_dict = dict(zip(word_idx, words))
    
    obj_idx = world[3][0] - 1
    objs = list(itertools.chain.from_iterable(map(list, world[4][0])))  
    obj_dict = dict(zip(obj_idx, objs))
    
    return word_dict, obj_dict

# %% [markdown]
# # Gold Standard Lexicon

# %%
def get_gold_lexicon(word_dict, obj_dict):
    
    lex = sio.loadmat('data/gold_standard.mat')['gold_standard']['map']
    lex = lex[0][0]
    
    lex_dict_index = {}
    lex_dict_readable = {}
    for i, j in zip(lex[0], lex[1]):
        lex_dict_index[i-1] = j-1
        lex_dict_readable[word_dict[i-1]] = obj_dict[j-1]
        
    return lex_dict_index, lex_dict_readable

# %% [markdown]
# # Corpus
# 

# %%
def get_corpus(word_dict, obj_dict):
    
    corpus = sio.loadmat('data/corpus.mat')['corpus']
    corpus = corpus[0]
    
    corpus_idx = []
    corpus_readable = []
    for pair in corpus:
        
        scene = list(pair[0][0]-1)
        utt = list(pair[1][0]-1)
        new_pair = {'scene': scene, 'utt': utt}
        corpus_idx.append(new_pair)
        
        scene_word = list(map(lambda x: obj_dict[x], scene))
        utt_word = list(map(lambda x: word_dict[x], utt))
        new_pair_readable = {'scene': scene_word, 'utt': utt_word}
        corpus_readable.append(new_pair_readable)
    
    return corpus_idx, corpus_readable

# %% [markdown]
# # Helper Functions

# %%
word_dict, obj_dict = get_world()
gold_idx, gold_readable = get_gold_lexicon(word_dict, obj_dict)
corpus_idx, corpus_readable = get_corpus(word_dict, obj_dict)

vocab_len = len(word_dict)
obj_len = len(obj_dict)


# %%
def get_alignments(pair):
    utt, scene = pair['utt'], pair['scene']
    combos = list(itertools.product(utt, scene))
    return combos


# %%
# def create_embeddings(vocab_len, obj_len):
#     # Access with torch.LongTensor()
#     # word_embeddings(torch.LongTensor([0, 1, 2])).shape = (3, 200)
#     word_embeddings = EmbeddingModeler(vocab_len, DIM_SIZE)
#     obj_embeddings = EmbeddingModeler(obj_len, DIM_SIZE)
#     return word_embeddings, obj_embeddings


# %%
class APEmbeddingModeler(nn.Module):

    def __init__(self, vocab_len, obj_len):
        super(APEmbeddingModeler, self).__init__()
        self.word_embeddings = nn.Parameter(torch.randn((vocab_len, DIM_SIZE)))
        self.object_embeddings = nn.Parameter(torch.randn((DIM_SIZE, obj_len)))
        self.parameters = nn.ParameterList([self.word_embeddings, self.object_embeddings])
    
    def forward(self, word, obj, neg_samples):
        word_embed = self.word_embeddings[word].view(1, -1) #1x200 and 200 by 50
        obj_embeds = F.cosine_similarity(word_embed, self.object_embeddings.t())
        obj_embed = obj_embeds[obj]
        neg_embeds = obj_embeds[neg_samples]

        #         word_embeds = self.word_embeddings(word).view((len(word), -1))
#         obj_embeds = self.object_embeddings(obj).view((len(obj), -1))
#         neg_sample_embeds = self.object_embeddings(neg_samples).view((len(neg_samples), -1))
        

        #TODO: PUT FORWARD ALL IN HERE, WHERE FORWARD IS JUST SELECTION OF EMBEDDINGs
        
        return word_embed, obj_embed, neg_embeds

class ASEmbeddingModeler(nn.Module):

    def __init__(self, vocab_len, obj_len):
        super(ASEmbeddingModeler, self).__init__()
        self.word_embeddings = nn.Parameter(torch.randn((DIM_SIZE, vocab_len)))
        self.object_embeddings = nn.Parameter(torch.randn(( obj_len, DIM_SIZE)))
        self.parameters = nn.ParameterList([self.word_embeddings, self.object_embeddings])
    
    def forward(self, word, obj, neg_samples):
        # word_embed = self.word_embeddings[word].view(1, -1) #1x200 and 200 by 50
        object_embed = self.object_embeddings[obj].view(1, -1) #1x200 and 200 by 50
        word_embeds = F.cosine_similarity(object_embed, self.word_embeddings.t())
        word_embed = word_embeds[word]
        # neg_embeds = obj_embeds[neg_samples]
        neg_embeds = word_embeds[neg_samples]

        #         word_embeds = self.word_embeddings(word).view((len(word), -1))
#         obj_embeds = self.object_embeddings(obj).view((len(obj), -1))
#         neg_sample_embeds = self.object_embeddings(neg_samples).view((len(neg_samples), -1))
        

        #TODO: PUT FORWARD ALL IN HERE, WHERE FORWARD IS JUST SELECTION OF EMBEDDINGs
        
        return word_embed, object_embed, neg_embeds




# %%
def train_embeddings(corpus_idx, corpus_readable, vocab_len, obj_len, loss_type, lr=0.001, epochs=10):
    
    if loss_type == 'anti-polysemy':
        model = APEmbeddingModeler(vocab_len, obj_len)
    elif loss_type == 'anti-synonymy':
        print("Using synonymy-loss")
        model = ASEmbeddingModeler(vocab_len, obj_len)
    
    losses = []
    optimizer = optim.SGD(model.parameters(), lr=lr) #TODO: MAKE ONE MODEL WITH BOTH EMBEDDINGS AS ELEMENTS
    for epoch in tqdm.tqdm_notebook(range(epochs)):
        print(f"On epoch {epoch}")
        total_loss = 0
        for i, pair in enumerate(corpus_idx):
            
#             readable = corpus_readable[i]
#             print(f"UTT: {pair['utt']}")
#             print(f"SCENE: {pair['scene']}")
#             print(f"UTT: {readable['utt']}")
#             print(f"SCENE: {readable['scene']}")
                  
            model.zero_grad()
            
            combos = get_alignments(pair)
            loss = 0
            for combo in combos:
                
                if loss_type == 'anti-polysemy':
                    loss += apl_loss(model, obj_len, combo, pair['scene'])
                elif loss_type == 'anti-synonymy':
                    loss += asl_loss(model, vocab_len, combo, pair['utt'])
                else:
                    loss += joint_loss(model, obj_len, vocab_len, combo)
                    
#                 print(loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

#             break
        losses.append(total_loss)
#         print(total_loss)
#     print(losses)  # The los
    
    return model, losses
    
    

# %% [markdown]
# # Neural Network Functions
# %% [markdown]
# INPUT needs to be a single world
# Run through an NN from 200 x H x 200
# First embedding space is word embedding
# Final embedding space is object embedding space.
# In this sense, the object embedding space is learned?
# Find the hinge loss between input word and chosen object, doesn't make sense??
# 
# Input should be a single word and a single referent.
# Then we need to find some relationship between them.
# Cosine should be that layer.
# Cosine should then lead to some scalar cos(w, o), which is input into a contrastive loss function with some other
# negative samples. Using backpropagation, all embeddings should then be learned.
# 
# 
# How does contrastive loss actually work?

# %%
def apl_loss(model, obj_len, combo, scene):
    
    word, obj = combo[0], combo[1]
    neg_samples = np.random.choice(np.setdiff1d(np.arange(obj_len), scene), NUM_NEG_SAMPLES, replace=False)

    word_embedding, obj_embedding, neg_sample_embeddings = model(word, obj, neg_samples)
    
    margin = 1 - obj_embedding +    neg_sample_embeddings
    
#     print("MARGIN")
#     print(margin)
    
    loss = torch.sum(
        torch.max(
            torch.zeros(NUM_NEG_SAMPLES),
            margin
        )
    )
#     print("LOSS")
#     print(loss)
    return loss

def asl_loss(model, vocab_len, combo, utt):
    word, obj = combo[0], combo[1]
    neg_samples = np.random.choice(np.setdiff1d(np.arange(vocab_len), utt), NUM_NEG_SAMPLES, replace=False)

    word_embedding, obj_embedding, neg_sample_embeddings = model(word, obj, neg_samples)
    
    margin = 1 - word_embedding +  neg_sample_embeddings
    
#     print("MARGIN")
#     print(margin)
    
    loss = torch.sum(
        torch.max(
            torch.zeros(NUM_NEG_SAMPLES),
            margin
        )
    )
#     print("LOSS")
#     print(loss)
    return loss


def joint_loss(word_embeddings, object_embeddings, obj_len, vocab_len):
    pol_loss = apl_loss()
    syn_loss = asl_loss()
    return torch.add(pol_loss, syn_loss)

def inv_dict(d):
    inv_map = {v: k for k, v in d.items()}
    return inv_map

def plot_embeds(model, word_dict, obj_dict, gold_lexicon):
    ########## plot all the object and word embeddings in the gold lexicon
    gold_ws = list(filter(lambda x: x in word_dict, set(gold_lexicon.keys())))
    gold_objs = list(filter(lambda x: x in obj_dict, set(gold_lexicon.values())))
    all_elems = gold_ws + gold_objs

    gold_w_embeds = np.array([model.word_embeddings[word_dict[w]].detach().numpy() for w in gold_ws])
    gold_obj_embeds = np.array([model.word_embeddings[obj_dict[o]].detach().numpy() for o in gold_objs])

    stacked_embeds = np.vstack((gold_w_embeds, gold_obj_embeds))
    t_embeds = TSNE(n_components=2, perplexity=10).fit_transform(stacked_embeds)
    t_w_embeds = t_embeds[0:len(gold_w_embeds)]
    t_obj_embeds = t_embeds[len(gold_w_embeds): ]

    fig, ax = plt.subplots(1,1)
    ax.scatter(t_w_embeds[:,0], t_w_embeds[:,1], c='r')

    for i in range(len(gold_ws)):
        ax.text(t_w_embeds[i,0], t_w_embeds[i, 1] , gold_ws[i] )

    for i in range(len(gold_objs)):
        ax.text(t_obj_embeds[i,0], t_obj_embeds[i, 1] , gold_objs[i] )

    ax.scatter(t_obj_embeds[:,0], t_obj_embeds[:,1], c='b')
    plt.savefig('blah.png')

    print(t_embeds)

def main(args):
    if args.train_model:
        all_losses = []
        # lrs = np.linspace(0.01, 0.9, 10)
        lrs = [0.9]
        epoch = 100
        for lr in tqdm.tqdm_notebook(lrs):
            print("training ")
            model, losses = train_embeddings(corpus_idx, corpus_readable, vocab_len, obj_len, 'anti-synonymy', lr, epoch)
            all_losses.append(losses)
        save_pkl('trained_model_syn', model)
        print("Saved!")



        # %%
        plt.plot(np.arange(100), all_losses[0])


        # %%
        # lrs = np.arange()
        # for i, losses in enumerate(all_losses):
        #     plt.plot(np.arange(10), losses)
        #     plt.title(f'Learning rate of {lrs[i]}')
        #     plt.show()
        fig, ax = plt.subplots(2, 5, figsize=(20, 10))

        all_losses = np.array(all_losses).reshape(2, 5, 100)
        for i in range(2):
            for j in range(5):
                ax[i][j].plot(np.arange(50), all_losses[i][j])
                ax[i][j].set_title(f'Learning rate of {lrs[5*i + j]}')
        plt.savefig('blah.png')

        # def plot_embeds(word_embeds, obj_embeds):
            # word embeds is of size |W| x 200
            # obj embeds is of size 200 x |O|
    elif args.inspect_model:
        print("inspecting")
        model = load_pkl('trained_model_syn')
        print(model)
        print(word_dict)
        print(obj_dict)
        print(gold_readable)
        plot_embeds(model, inv_dict(word_dict), inv_dict(obj_dict), gold_readable)

# %%



# %% [markdown]
# # Framework

# %%



parser = argparse.ArgumentParser()
parser.add_argument('--train_model', action='store_true')
parser.add_argument('--inspect_model', action='store_true')
main(parser.parse_args())