from path import *
import numpy as np
from numpy import dot
from numpy.linalg import norm
import nltk
import pandas as pd
from frame_similarity import get_frames_count

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy import spatial


# Remember to set path to word2vec model
path_to_model = os.path.join(data_path, "mymodel.gsm")
model = Word2Vec.load(path_to_model)
word_vectors = model.wv

def load_dataframe(file = os.path.join(data_path,"data.pkl")) :
    return (pd.read_pickle(file))

# Computes the number elements in L and M which are non-zero at the same time
def element_similarity(L,M):
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])

def normalized_cosine_sim(L,M) :
    if min(sum(L),sum(M)) == 0 :
        return 0
    l = [elt/sum(L) for elt in L]
    m = [elt/sum(M) for elt in M]
    return np.dot(l,m)/(np.linalg.norm(l)*np.linalg.norm(m))

######################### Dice-coefficient ##########################
# from source https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Dice%27s_coefficient
def dice_coefficient(a,b):
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """

    if a == b: return 1.0

    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0

    a_bigram_list = [a[i:i+2] for i in range(len(a)-1)]
    b_bigram_list = [b[i:i+2] for i in range(len(b)-1)]

    a_bigram_list.sort()
    b_bigram_list.sort()

    # assignments to save function calls
    len_a = len(a_bigram_list)
    len_b = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < len_a and j < len_b):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1

    score = float(matches)/float(len_a + len_b)
    return score

######################### tf-idf measures ##########################

def tfidf_vector_similarity(sentence_1, sentence_2):
    corpus = [sentence_1, sentence_2]
    vectorizer = TfidfVectorizer(min_df=1)
    vec_1 = vectorizer.fit_transform(corpus).toarray()[0]
    vec_2 = vectorizer.fit_transform(corpus).toarray()[1]
    cos_sim = np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return cos_sim


######################### jaccard coefficient ##########################

def jaccard_sim_coefficient(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    intersection_words = set(words_1).intersection(set(words_2))
    return len(intersection_words)/len(joint_words)

######################### cosine measures ##########################

def avg_sentence_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])

    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def embedding_sentence_similarity(sentence_1, sentence_2):
    sent1 = sentence_1.split()
    sent2 = sentence_2.split()
    index2word_set = set(word_vectors.index2word)
    sent1_avg = avg_sentence_vector(sent1, model, 100, index2word_set)
    sent2_avg = avg_sentence_vector(sent2, model, 100, index2word_set)
    similarity = 1 - spatial.distance.cosine(sent1_avg, sent2_avg)
    return similarity

######################### word mover distance measure ##########################

def WMD(sentence_1, sentence_2):
    words_1 = sentence_1.split()
    words_2 = sentence_2.split()
    distance = word_vectors.wmdistance(words_1, words_2)
    similarity = 1 - distance
    return similarity


""" Given a document id, returns the topn most similar documents in terms of
the cosine similarity between their respective frame counts. The vectors are normalized
beforehand. """

def most_similar(id1, topn = 50, df = pd.DataFrame()) :
    if len(df) == 0 :
        df = load_dataframe()

    L = get_frames_count(id1, df)

    def score(id2) :                   # inner function, returns the similarity between id1 and id2
        M = get_frames_count(id2, df)
        return normalized_cosine_sim(L,M)

    docs = [df.index[i] for i in range(len(df)) if df.index[i]!= id1]
    
    return (sorted(docs, key = score, reverse = True)[:topn])
