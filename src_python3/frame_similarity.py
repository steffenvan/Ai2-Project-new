import pandas as pd
from extraction import *
from path import *

# This computes the similarity of the common frames of an abstract with all
# the other abstracts we have.

# Remember to set the correct path
json_path = os.path.join(parent + "/json_abs/")

df = pd.read_pickle("data.pkl")
reference_file = json.load(open(json_path + str(df.index[0])))
all_other_abstracts = df.index[1:]

# Given two lists of words, computes how many words appear in both lists
def identical_words(L,M):
    count = 0

    if len(M) > len(L):
        for elt in L :
            if elt in M :
                count += 1
    else:
        for elt in M :
            if elt in L :
                count += 1
    return count

def get_line(df, id) :
    return (df.loc[id,:]).tolist()

reference = get_line(df, df.index[0])

# Computes the number elements in L and M which are non-zero at the same time
def element_similarity(L,M):
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])

# Returns a dictionary of the frames of a file.
# Keys: Frames
# Values: Corresponding text to the frame
def tokenize_relevant_frames(file):
    text_dict = {}
    for sentence in file:
        for frame in sentence["frames"]:
            if frame["target"]["name"] in df.columns:
                result = extract_text(frame)
                annot = " ".join(result["annot"])
                nlp_tokens = nlp(annot)
                text_dict.update({result["name"] : nlp_tokens})
                # print("Frame: %s, Target: %s" % (result["name"], result["target"]))
                # print("Full text: ", nlp_tokens)
    return text_dict

reference_file_result = tokenize_relevant_frames(reference_file)

# Computes the similarity value between the common frames of the reference abstract
# and all the other abstracts one by one.
# TODO: find a proper value for the threshold (elem_sim).

for abs_id in all_other_abstracts:
    elem_sim = element_similarity(reference, get_line(df, abs_id))
    if elem_sim > 4 :
        print("****************************************************************\n")
        print(abs_id)
        temp_abs = json.load(open(json_path + str(abs_id)))
        temp_file_result = tokenize_relevant_frames(temp_abs)

        # Find common frames between the reference and other abstracts.
        for key in reference_file_result.keys() & temp_file_result.keys():
            simil_val = reference_file_result[key].similarity(temp_file_result[key])
            print("Similar frame: %s" % key)
            print("Similarity value: %f \nReference text: %s \nCompared text: %s\n" % (simil_val,
                                                                                       reference_file_result[key],
                                                                                       temp_file_result[key]))
