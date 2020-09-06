import argparse
from torch import randn, tensor
import scipy.io as sio
import pandas as pd
import numpy as np
import itertools

from torch.utils.data import SequentialSampler
from torch import stack
from torch.nn import MarginRankingLoss, CosineSimilarity

def get_world():
    
    world = sio.loadmat('data/world.mat')['world']
    world = world[0][0]
    
    word_idx = world[0][0]
    words = list(itertools.chain.from_iterable(map(list, world[1][0])))  
    word_dict = dict(zip(word_idx, words))
    
    obj_idx = world[3][0]
    objs = list(itertools.chain.from_iterable(map(list, world[4][0])))  
    obj_dict = dict(zip(obj_idx, objs))
    return word_dict, obj_dict



def get_gold_lexicon(word_dict, obj_dict):
    
    lex = sio.loadmat('data/gold_standard.mat')['gold_standard']['map']
    lex = lex[0][0]
    
    lex_dict_index = {}
    lex_dict_readable = {}
    for i, j in zip(lex[0], lex[1]):
        lex_dict_index[i] = j
        lex_dict_readable[word_dict[i]] = obj_dict[j]
        
    return lex_dict_index, lex_dict_readable



def get_corpus(word_dict, obj_dict):
    
    corpus = sio.loadmat('data/corpus.mat')['corpus']
    corpus = corpus[0]
    
    corpus_idx = []
    corpus_readable = []
    for pair in corpus:
        
        scene = list(pair[0][0])
        utt = list(pair[1][0])
        new_pair = {'scene': scene, 'utt': utt}
        corpus_idx.append(new_pair)
        
        scene_word = list(map(lambda x: obj_dict[x], scene))
        utt_word = list(map(lambda x: word_dict[x], utt))
        new_pair_readable = {'scene': scene_word, 'utt': utt_word}
        corpus_readable.append(new_pair_readable)
    
    #print(corpus_readable)
    return corpus_idx, corpus_readable

def get_embeds(vocab, objs):
    assert len(vocab) == len(set(vocab)) # ensure vocab is unique words
    vocab_embeds = randn(len(vocab), 50) 
    obj_embeds = randn(len(objs), 50)
    # types = ['word' for word in vocab] + ['obj' for obj in objs]

    # frame = pd.DataFrame({'embed':embeds}, index = pd.Index(vocab + objs, name='item'))
    # frame['type'] = types
    create_map = lambda ws, emb_list: {ws[i]: emb_list[i] for i in range(len(ws))}
    return create_map(vocab, vocab_embeds), create_map(objs, obj_embeds)

def upperize(strs):
    return list(map(lambda x: x.upper(), strs))

def train_loop(w_to_embeds, obj_to_embeds, corpus):
    num_epochs = 20 # number of sweeps through corpus
    product = itertools.product
    get_key_random_sample = lambda d: np.random.choice(np.array(list(d.keys())))
    cos = CosineSimilarity(dim=1)
    loss_f = MarginRankingLoss() 
    for scene_utt_pair_d in corpus:
        scene_objs = upperize(scene_utt_pair_d['scene'])
        scene_utter = scene_utt_pair_d['utt']

        loss = 0
        for (obj, word) in product(scene_objs, scene_utter):
            # compute loss 
            obj_embed = obj_to_embeds[obj]
            word_embed = w_to_embeds[word]
            neg_sample_w = get_key_random_sample(w_to_embeds)
            neg_sample_embed = w_to_embeds[neg_sample_w]
            cos_vals = cos(stack((obj_embed, obj_embed)), stack((word_embed, neg_sample_embed)))
            y = tensor([1, -1])
            loss += loss_f(cos_vals, y)
    return 
    
if __name__ == '__main__':
    word_dict, obj_dict = get_world()
    objs = upperize(list(obj_dict.values()))
    w_to_embs, obj_to_embs  = get_embeds(list(word_dict.values()), objs)
    corpus_idx, corpus_readable = get_corpus(word_dict, obj_dict)
    train_loop(w_to_embs, obj_to_embs, corpus_readable)
    gold_idx, gold_readable = get_gold_lexicon(word_dict, obj_dict)