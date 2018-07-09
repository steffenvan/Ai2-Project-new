# This computes the similarity of the common frames of an abstract with all
# the other abstracts we have.

import pandas as pd
from extraction import *
from path import *
import operator
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Remember to set the correct path
json_path = os.path.join(parent + "/json_abs/")

# Opening correct files
df = pd.read_pickle("data.pkl")
reference_file = json.load(open(json_path + str(df.index[0])))
all_other_abstracts = df.index[1:]

def get_frames_count(df, id) :
    res = (df.loc[id,:]).tolist()
    return res

reference_frames = get_frames_count(df, df.index[0])

# Computes the number elements in L and M which are non-zero at the same time
def element_similarity(L,M):
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])

# Returns a dictionary of the frames of a file.
# Keys: Frames
# Values: Full sentence in which the frames are found

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
    lena = len(a_bigram_list)
    lenb = len(b_bigram_list)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a_bigram_list[i] == b_bigram_list[j]:
            matches += 2
            i += 1
            j += 1
        elif a_bigram_list[i] < b_bigram_list[j]:
            i += 1
        else:
            j += 1

    score = float(matches)/float(lena + lenb)
    return score

######################### tf-idf measures ##########################

def tfidf_vector_similarity(sentence_1, sentence_2):
    corpus = [sentence_1, sentence_2]
    vectorizer = TfidfVectorizer(min_df=1)
    vec_1 = vectorizer.fit_transform(corpus).toarray()[0]
    vec_2 = vectorizer.fit_transform(corpus).toarray()[1]
    sim = np.dot(vec_1, vec_2.T) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))
    return sim

######################### jaccard coefficient ##########################
def jaccard_similarity_coefficient(sentence_1, sentence_2):
    words_1 = nltk.word_tokenize(sentence_1)
    words_2 = nltk.word_tokenize(sentence_2)
    joint_words = set(words_1).union(set(words_2))
    intersection_words = set(words_1).intersection(set(words_2))
    return len(intersection_words)/len(joint_words)

def extract_frame_sentence(file):
    sentence = ""
    frame_and_sentence = {}
    for list_of_frames in file:
        for frame in list_of_frames["frames"]:
            if frame["target"]["name"] in df.columns:
                sentence = " ".join(list_of_frames["tokens"])
                frame_and_sentence.update({frame["target"]["name"] : sentence})

    return frame_and_sentence

ref_frame_and_sentence = extract_frame_sentence(reference_file)

# Computes the similarity value of the full sentence of the common frames of the reference abstract
# and all the other abstracts one by one.
# TODO: find a proper value for the threshold (elem_sim).

# Dictionary for the paper and its similarity value with the reference paper.
paper_and_cos_val = {}
paper_and_dice_val = {}
paper_and_jaccard_val = {}

tfidf_vectorizer = TfidfVectorizer()

for abstract_id in all_other_abstracts:
    sim_frame_count = element_similarity(reference_frames, get_frames_count(df, abstract_id))
    total_sim_val = 0.0
    total_dice_val = 0.0
    total_jaccard_val = 0.0
    #total_sent_val = 0.0
    if sim_frame_count > 4 :
        print("\n****************************************************************\n")
        print(abstract_id, "\n")
        temp_abstract = json.load(open(json_path + str(abstract_id)))
        # temp_result = tokenize_relevant_frames(temp_abstract)
        temp_frame_and_sentence = extract_frame_sentence(temp_abstract)

        # Find common frames between the reference and other abstracts.
        for key in ref_frame_and_sentence.keys() & temp_frame_and_sentence.keys():
            # Resetting similarity value for each frame
            cos_sim = 0.0
            dice_sim = 0.0
            jaccard_sim = 0.0
            #sent_sim = 0.0

            #sentences_tuple = (ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            #sentence_matrix = tfidf_vectorizer.fit_transform(sentences_tuple)

            # Cosine similarity
            cos_sim = tfidf_vector_similarity(ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            total_sim_val += cos_sim
            # dice similarity
            dice_sim = dice_coefficient(ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            total_dice_val += dice_sim

            # jaccard similarity
            jaccard_sim = jaccard_similarity_coefficient(ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            total_jaccard_val += jaccard_sim

            # other similarity
            #info_content_norm = True
            #sent_sim = similarity(ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            #total_sent_val += sent_sim

            paper_and_cos_val.update({abstract_id:total_sim_val})
            paper_and_dice_val.update({abstract_id:total_dice_val})
            paper_and_jaccard_val.update({abstract_id:total_jaccard_val})

            print("Similar frame: %s" % key)
            print("Cosine similarity: %f" % cos_sim)
            print("Dice coefficient: %f" % dice_sim)
            print("Jaccard similarity: %f" % jaccard_sim)
            #print("Sentence similarity: %f" % sent_sim)
            print("Reference text: %s" % ref_frame_and_sentence[key])

            print("Compared text: %s\n" % temp_frame_and_sentence[key])
        print("Total cos value: %f" % total_sim_val)
        print("Total dice val: %f" % total_dice_val)
        print("Total jaccard val: %f" % total_jaccard_val)
        #print("Total sentence val: %f" % total_sent_val)

#Ranking the papers from highest to lowest
sorted_cos          = sorted(paper_and_cos_val.items(), key=lambda x: x[1], reverse=True)
sorted_dice         = sorted(paper_and_dice_val.items(), key=lambda x: x[1], reverse=True)
sorted_jaccard      = sorted(paper_and_jaccard_val.items(), key=lambda x: x[1], reverse=True)
best_paper_cos      = max(paper_and_cos_val.items(), key=operator.itemgetter(1))[0]
best_paper_dice     = max(paper_and_dice_val.items(), key=operator.itemgetter(1))[0]
best_paper_jaccard  = max(paper_and_jaccard_val.items(), key=operator.itemgetter(1))[0]

print("\nPaper and Cos values:\n", sorted_cos)
print("\nPaper and Dice values:\n", sorted_dice)
print("\nPaper and Jaccard values:\n", sorted_jaccard)
print("\nSimilar paper count:", len(paper_and_cos_val))
print("\nMost similar paper using cosine:", best_paper_cos)
print("\nMost similar paper using dice:", best_paper_dice)
print("\nMost similar paper using jaccard:", best_paper_jaccard)
