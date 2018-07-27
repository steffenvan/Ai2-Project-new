import scipy.sparse
import os
from pathlib import Path
import sys
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
import pickle
import bisect
import time

def load_tfidf(path = os.path.join(data_path, "tfidf.pkl")) :
    with open(path, "rb") as f :
        object = pickle.load(f)
    return object

def load_tfidf_bigrams(path = os.path.join(data_path, "tfidf2grams.pkl")) :
        with open(path, "rb") as f :
            object = pickle.load(f)
        return object 
        
def load_tfidf_trigrams(path = os.path.join(data_path, "tfidf3grams.pkl")) :
            with open(path, "rb") as f :
                object = pickle.load(f)
            return object

def tfidf_value(word, file, map_file, vocabulary, X) :
    try :
        file_index = map_file[file]
        word_index = vocabulary[word]
        value = X[file_index, word_index]
        return value
    except :
        return 0

def highest_tfidf_words(doc_id, map_file, vocabulary, X, nmax = 20) :
    tuples = []
    file_index = map_file[doc_id]
    txt = open(get_path(doc_id, False, True)).read()
    unique_words = []
    for word in txt.split(" ") :
        word = word.lower()
        if word not in unique_words :
            unique_words.append(word)
            if word in vocabulary :
                word_index = vocabulary[word]
                # tuples.append((word,X[file_index, word_index]))
                bisect.insort_right(tuples, (X[file_index, word_index], word))
    return tuples[-nmax:]
    


def to_bigrams(sentence) :
    sentence = sentence.split(" ")
    return [sentence[i] + " "+ sentence[i+1] for i in range(len(sentence)-1)]

def to_trigrams(sentence) :
    sentence = sentence.split(" ")
    return [sentence[i] + " " + sentence[i+1] + " " + sentence[i+2] for i in range(len(sentence)-2)]
    

# def highest_tfidf_bigrams(doc_id, map_file, vocabulary, X, nmax = 20) :
#     tuples = []
#     file_index = map_file[doc_id]
#     txt = open(get_path(doc_id, False, True)).read()
#     unique_bigrams = []
#     for sent in txt.split('\n') :
#         for bigram in to_bigrams(sent) :
#             bigram = bigram.lower()
#             if bigram not in unique_bigrams :
#                 unique_bigrams.append(bigram)
#                 if bigram in vocabulary :
#                     bg_index = vocabulary[bigram]
#                     # tuples.append((word,X[file_index, word_index]))
#                     bisect.insort_right(tuples, (X[file_index, bg_index], bigram))
#     return tuples[-nmax:]
    

def highest_tfidf_bigrams(doc_id, map_file, vocabulary, X, nmax = 20) :
    tuples = []
    file_index = map_file[doc_id]
    txt = open(get_path(doc_id, False, True)).read()
    unique_bigrams = []
    for sent in txt.split('\n') :
        for bigram in to_bigrams(sent) :
            bigram = bigram.lower()
            if bigram not in unique_bigrams :
                unique_bigrams.append(bigram)
                if bigram in vocabulary :
                    bg_index = vocabulary[bigram]
                    tuples.append((bigram, X[file_index, bg_index]))
    tuples = sorted(tuples, reverse = True, key=lambda tup: tup[1])
    return tuples[:nmax]    
    
# [vectorizer, X, map_file] = load_tfidf_bigrams()
# vocabulary = vectorizer.vocabulary_
# print(highest_tfidf_bigrams2('E09-1012-parscit.130908', map_file, vocabulary, X, nmax = 50))

def highest_tfidf_trigrams(doc_id, map_file, vocabulary, X, nmax = 20) :
    tuples = []
    file_index = map_file[doc_id]
    txt = open(get_path(doc_id, False, True)).read()
    unique_trigrams = []
    for sent in txt.split('\n') :
        for trigram in to_trigrams(sent) :
            trigram = trigram.lower()
            if trigram not in unique_trigrams :
                unique_trigrams.append(trigram)
                if trigram in vocabulary :
                    tg_index = vocabulary[trigram]
                    # tuples.append((word,X[file_index, word_index]))
                    bisect.insort_right(tuples, (X[file_index, tg_index], trigram))
    return tuples[-nmax:]

# print(tfidf_value("anaphora",'D15-1260-parscit.130908.txt', map_file, vocabulary, X))
# [vectorizer, X, map_file] = load_tfidf_trigrams()
# vocabulary = vectorizer.vocabulary_
# # print(len(vocabulary))
# # beg = time.time()
# print(highest_tfidf_trigrams('P09-4009-parscit.130908', map_file, vocabulary, X, nmax = 3))
# # print(tfidf_value("anaphora",'D15-1260-parscit.130908.txt', map_file, vocabulary, X))
# print(time.time() - beg)
# test_file = 'D15-1260-parscit.130908.txt'
# 
# L = []
# d = {}

# def extract_relevant_sentences()
# for line in open(os.path.join(txt_train_path, test_file)).readlines() :
#         mean = 0
#         word_counter = 0
#         for word in line.split(" ") :
#             try :
#                 mean += tfidf_value(word.lower(), test_file, map_file, vocabulary, X)
#                 word_counter += 1.0
#             except :
#                 # print("error for " + word)
#                 a = 0
#         if word_counter == 0 :
#             mean = 0
#         else :
#             mean /= word_counter
#         d.update({mean : line})
# 
# keylist = d.keys()   
# for key in sorted(list(d.keys), reverse = True)[:50] :
#     L.append((d[key],key))
# 
# print(L)




















