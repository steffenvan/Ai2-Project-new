# This computes the similarity of the common frames of an abstract with all
# the other abstracts we have.

import pandas as pd
from extraction import *
from path import *
import operator

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
paper_and_sim_val = {}
tfidf_vectorizer = TfidfVectorizer()

for abstract_id in all_other_abstracts:
    sim_frame_count = element_similarity(reference_frames, get_frames_count(df, abstract_id))
    total_sim_val = 0.0
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

            sentences_tuple = (ref_frame_and_sentence[key], temp_frame_and_sentence[key])
            sentence_matrix = tfidf_vectorizer.fit_transform(sentences_tuple)

            cos_sim = cosine_similarity(sentence_matrix[0:1], sentence_matrix)[0][1]
            total_sim_val += cos_sim

            paper_and_sim_val.update({abstract_id:total_sim_val})

            print("Similar frame: %s" % key)
            print("Cosine similarity: %f" % cos_sim)
            print("Reference text: %s" % ref_frame_and_sentence[key])

            print("Compared text: %s\n" % temp_frame_and_sentence[key])
        print("Total value: %f" % total_sim_val)

# Ranking the papers from highest to lowest
sorted_values = sorted(paper_and_sim_val.items(), key=lambda x: x[1], reverse=True)
print(sorted_values)
best_paper    = max(paper_and_sim_val.items(), key=operator.itemgetter(1))[0]
print("Similar paper count:", len(paper_and_sim_val))
print("\nMost similar paper:", best_paper)
