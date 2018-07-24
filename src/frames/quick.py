import os
from pathlib import Path
import sys
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
from tfidf_functions import *
from file_content_extraction import *
from gensim.models import Word2Vec
from distance_measures import *

object = load_tfidf()
vectorizer = object[0]
X = object[1]
map_file = object[2]
vocabulary = vectorizer.vocabulary_
wv_model = Word2Vec.load(str(path_to_model))

test_file = 'D15-1260-parscit.130908.json'


# doc_id = os.path.join(json_train_path, test_file)
# text_id = os.path.join(txt_train_path, 'D15-1260-parscit.130908.txt')

d = important_sents_by_frame("D15-1260-parscit.130908", True)   

sentence = d['Position_on_a_scale'][0]

print(sent_to_vec(sentence, "D15-1260-parscit.130908", map_file, vocabulary, X, wv_model))