from pathlib import Path
import sys
import os
# Setting the correct path to retrieve path.py
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
from math import sqrt, isnan
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy import spatial
from tfidf_functions import *
# Remember to set path to word2vec model. Using os.path.join because of a bug
# in posixpath.
# path_to_model = os.path.join(data_path, "mymodel.gsm")
# model = Word2Vec.load(path_to_model)
# word_vectors = model.wv


######################### Dice-coefficient ##########################
def dice_coefficient(sentence_1, sentence_2):
    if not len(sentence_1) or not len(sentence_2): return 0.0
    """ quick csentence_1se for true duplicates """

    if sentence_1 == sentence_2: return 1.0

    """ if sentence_1 != sentence_2, and sentence_1 or sentence_2 are single chars, then can't possibly match """
    if len(sentence_1) == 1 or len(sentence_2) == 1: return 0.0

    sentence_1_bigram_list = [sentence_1[i:i+2] for i in range(len(sentence_1)-1)]
    sentence_2_bigram_list = [sentence_2[i:i+2] for i in range(len(sentence_2)-1)]

    sentence_1_bigram_list.sort()
    sentence_2_bigram_list.sort()

    # assignments to save function calls
    len_sentence_1 = len(sentence_1_bigram_list)
    len_sentence_2 = len(sentence_2_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < len_sentence_1 and j < len_sentence_2):
        if sentence_1_bigram_list[i] == sentence_2_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif sentence_1_bigram_list[i] < sentence_2_bigram_list[j]:
            i += 1
        else:
            j += 1

    score = float(matches)/float(len_sentence_1 + len_sentence_2)
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
    
def sent_to_vec(sentence, file, map_file, vocabulary, X, wv_model) :
    words = sentence.split(" ")
    tfidfs = [tfidf_value(word.lower(), file, map_file, vocabulary, X) for word in words if word.lower() in vocabulary]
    sentence_vector = np.zeros(100)
    if len(tfidfs) == 0 :
        return sentence_vector
    # trigger = sum(tfidfs)/len(tfidfs)
    trigger = 0.2
    # print(trigger)
    i = 0
    
    for word in words :
        word = word.lower()
        if word in vocabulary and word in wv_model :

            word_tfidf = tfidf_value(word, file, map_file, vocabulary, X)
            if word_tfidf > trigger :
            # print(word_tfidf)
                i += 1
                word_vector = wv_model[word]
                sentence_vector = np.add(sentence_vector, [word_tfidf*coord for coord in word_vector])
    if i == 0 :
        return sentence_vector
    return [coord/i for coord in sentence_vector]
            
            
    

def embedding_sentence_similarity(sentence_1, sentence_2):
    sent1 = sentence_1.split()
    sent2 = sentence_2.split()
    index2word_set = set(word_vectors.index2word)
    sent1_avg = avg_sentence_vector(sent1, model, 100, index2word_set)
    sent2_avg = avg_sentence_vector(sent2, model, 100, index2word_set)
    similarity = 1 - spatial.distance.cosine(sent1_avg, sent2_avg)
    return similarity

def normalized_cosine_sim(L,M) :
    if min(sum(L),sum(M)) == 0 :
        return 0

    l = [elt/sum(L) for elt in L]
    m = [elt/sum(M) for elt in M]
    value = np.dot(l,m)/(np.linalg.norm(l)*np.linalg.norm(m))
    if isnan(value) :
        return 0 
    return value

######################### word mover distance measure ##########################

def WMD(sentence_1, sentence_2):
    words_1 = sentence_1.split()
    words_2 = sentence_2.split()
    distance = word_vectors.wmdistance(words_1, words_2)
    similarity = 1 - distance
    return similarity

def cosine_sim(L, M) :
    return np.dot(L,M)/(np.linalg.norm(L)*np.linalg.norm(M))