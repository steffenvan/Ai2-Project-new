import scipy.sparse
import os
from pathlib import Path
import sys
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
import pickle

def load_tfidf(path = os.path.join(data_path, "tfidf.pkl")) :
    with open(path, "rb") as f :
        object = pickle.load(f)
    return object
    
# object = load_tfidf()
# vectorizer = object[0]
# X = object[1]
# map_file = object[2]
# vocabulary = vectorizer.vocabulary_
# print(map_file['D15-1260-parscit.130908.txt'])

def tfidf_value(word, file, map_file, vocabulary, X) :
    file_index = map_file[file]
    word_index = vocabulary[word]
    # print(file_index)
    # print(word_index)
    try :
        return X[file_index, word_index]
    except :
        return 0
        
# print(tfidf_value("anaphora",'D15-1260-parscit.130908.txt', map_file, vocabulary, X))

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




















