import gensim
import os
import sys
import pandas as pd
import pickle
from numpy import dot
from numpy.linalg import norm
from random import shuffle
from path import *

def cosine_sim(L, M) :
    return dot(L,M)/(norm(L)*norm(M))

model = gensim.models.Word2Vec.load(os.path.join(data_path,"mymodel.gsm"))

txt_folder = os.path.join(data_path,"txt/")


def sentence_vectorize(sentence, important_words) :
    vec = 100*[0]
    l = 0
    for word in sentence.split(" ") :
        if word in model.wv.vocab :
            vec += model[word]*(1 + 0.5*int(word in important_words))
            l += 1 + 0.5 * int(word in important_words)
    if l == 0 :
        return 100*[0]
    return [elt/l for elt in vec]


#
# file_list = os.listdir(txt_folder)
# shuffle(file_list)
#
# for filename in file_list[:500] :
#     print(filename)
#     if filename.endswith("txt") :
#         file = os.path.join(txt_folder, filename)
#         for sentence in open(file).read().split("\n") :
#             sent_vec = sentence_vec(sentence.lower())
#             if norm(sent_vec) > 0 and cosine_sim(input_vec, sent_vec) > 0.8 :
#                 print(sentence)
#                 print(cosine_sim(input_vec, sent_vec))
