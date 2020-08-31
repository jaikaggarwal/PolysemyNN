import scipy.io as sio
import pandas as pd
import numpy as np
import itertools

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
    
    return corpus_idx, corpus_readable

if __name__ == '__main__':
	word_dict, obj_dict = get_world()
	gold_idx, gold_readable = get_gold_lexicon(word_dict, obj_dict)
	corpus_idx, corpus_readable = get_corpus(word_dict, obj_dict)