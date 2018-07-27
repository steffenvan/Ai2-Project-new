from pathlib import Path
import sys
import os
import scipy
# Setting the correct path to retrieve path.py
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
from file_content_extraction import *
from distance_measures import *
from similarity_utility import *
import pickle
from itertools import chain



train_files = os.listdir(txt_train_path) 
test_files = os.listdir(txt_test_path)

file_objects = chain((open(os.path.join(txt_train_path, file)).read() for file in train_files if file.endswith("txt")), (open(os.path.join(txt_test_path, file)).read() for file in test_files if file.endswith("txt")))  # generator to avoid loading everything in memory at once

file_names = [file for file in train_files if file.endswith("txt")] + [file for file in test_files if file.endswith("txt")]

i = 0
map_file = {}
for file in file_names :
    map_file.update({file[:-4] : i})
    i += 1
    
vectorizer = TfidfVectorizer(stop_words = "english", ngram_range=(3,3))

X = vectorizer.fit_transform(file_objects)

object = [vectorizer, X, map_file]
    
with open(os.path.join(data_path, "tfidf3grams.pkl"), "wb+") as f :
    pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

# with open(os.path.join("tfidf.pkl"), "rb") as f2 :
#     object2 = pickle.load(f2)
    
# for elt in object2 :
#     print((elt))
    
    
# 
# example_file = 'D15-1260-parscit.130908.txt'

# print(vectorizer.vocabulary_["chunking"])
# print(X[map_file[example_file],vectorizer.vocabulary_["anaphora"]])
# d = {}
# inv_vocab = vectorizer.get_feature_names()
# for word_index in X[map_file[example_file],:] :
#     value = X[map_file[example_file],word_index]
#     if value != 0 :
#         d.update({inv_vocab[word_index] : value} )
# print(d)

# scipy.sparse.save_npz('/Users/paulazoulai/Desktop/little_sparse_matrix.npz', X)
# import pickle
# 
# filehandler = open("/Users/paulazoulai/Desktop/test.pkl", 'w+')
# pickle.dump(X, filehandler)
# docs = [doc1, open("/Users/paulazoulai/Desktop/pre/data/train/txt/W03-1015-parscit.130908.txt", "r")]
# 
# vectorizer = TfidfVectorizer() #stop_words='english'
# X = vectorizer.fit_transform(docs)
# print(vectorized_abs)
# words = np.array(vectorizer.get_feature_names())
# 
# print(vectorizer.get_feature_names())




