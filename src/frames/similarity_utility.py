from path import *
import numpy as np
from numpy import dot
from numpy.linalg import norm
import nltk
import pandas as pd
# Setting the correct path to retrieve path.py
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from scipy import spatial

# Remember to set path to word2vec model
path_to_model = data_path.joinpath("mymodel.gsm")
model = Word2Vec.load(path_to_model)
word_vectors = model.wv


# Load the dataframe for easy use.
def load_dataframe(file = os.path.join(data_path,"data.pkl")) :
    return (pd.read_pickle(file))

# Also adding a weight the important words.
# TODO: experiment with the weight of the important words
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

def get_frames_count(id, df = pd.DataFrame()) :
    if len(df) == 0 :
        df = load_dataframe()
    res = (df.loc[id,:]).tolist()
    return res

# Computes the number of similar frames. Finds the similar frames for two articles.
def common_frames(L,M):
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])

def normalized_cosine_sim(L,M) :
    if min(sum(L),sum(M)) == 0 :
        return 0

    l = [elt/sum(L) for elt in L]
    m = [elt/sum(M) for elt in M]

    return np.dot(l,m)/(np.linalg.norm(l)*np.linalg.norm(m))

######################### Dice-coefficient ##########################
def dice_coefficient(sentence_1, sentence_2):
    if not len(sentence_1) or not len(sentence_2): return 0.0
    """ quick csentence_1se for true duplicates """

    if sentence_1 == sentence_2: return 1.0

    """ if sentence_1 != sentence_2, and sentence_1 or sentence_2 are single chars, then they can't possibly match """
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
