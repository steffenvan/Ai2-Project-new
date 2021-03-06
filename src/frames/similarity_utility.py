from pathlib import Path
import sys
import pandas as pd
# Setting the correct path to retrieve path.py
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from distance_measures import normalized_cosine_sim

from path import *

# File loading utilities
def get_document_text(document_id):
    format = "txt"
    file_path = txt_train_path.joinpath(str(document_id[:-4]) + format)
    content = Path(file_path).read_text()
    return content

# Load the dataframe for easy use.
def load_dataframe(file = data_path.joinpath("train_data.pkl")) :
    return (pd.read_pickle(file))

def load_tfidf_dataframe(file = data_path.joinpath("train_tfidf.pkl")) :
    return (pd.read_pickle(file))

# Also adding a weight the important words.
# TODO: experiment with the weight of the important words
def sentence_vectorize(sentence, important_words, model) :
    # print("TESTTTTTT")
    vec = 100*[0]
    l = 0
    for word in sentence.split(" ") :
        if word in model.wv.vocab :
            vec += model[word]*(1 + 0.5*int(word in important_words))
            l += 1 + 0.5 * int(word in important_words)
    if l == 0 :
        
        return 100*[0]
    
    return [elt/l for elt in vec]

# Computes the number of a particular frame in a document.
def get_frames_count(id, df) :
    res = (df.loc[id,:]).tolist()
    return res

def get_doc_bigrams(id, df) :
    res = (df.loc[id,:]).tolist()
    return res
    
def intersect(L,M) :
    return [elt for elt in list(L) if elt in list(M)]
    return [(i,word) for i,word in enumerate(list(L)) if word in M]

# Computes the number of similar frames. Finds the similar frames for two articles.
def common_frames(L,M):
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])

""" Given a document id, returns the topn most similar documents in terms of
cosine similarity between their respective frame counts. The vectors are normalized beforehand. """

def most_similar(id1, df, topn = 50) :
    L = get_frames_count(id1, df)
    def score(id2) :                   # inner function, returns the similarity between id1 and id2
        M = get_frames_count(id2, df)
        return normalized_cosine_sim(L,M)
        
    docs = [df.index[i] for i in range(len(df)) if df.index[i]!= id1]
    most_sim_docs = sorted(docs, key = score, reverse = True)[:topn]
    return most_sim_docs
    
def most_similar_with_bigrams(id1, df, topn = 50) :
    L = get_doc_bigrams(id1, df)
    def score(id2) :
        M = get_doc_bigrams(id2, df)
        s = len(intersect(L,M))
        if s > 0 :
            print(s)
        return s
    docs = [df.index[i] for i in range(len(df)) if df.index[i]!= id1]
    most_sim_docs = sorted(docs, key = score, reverse = True)[:topn]
    return most_sim_docs