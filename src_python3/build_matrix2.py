from path import *
from extraction import *
import os
import pandas as pd
import spacy


json_path = os.path.join(parent + "/json_abs/")


frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']

def build_matrix() :
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)
    ids = [filename for filename in os.listdir(json_path) if filename.endswith("json")]
    df["ID"] = ids
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        file = open(str(json_path+"/"+ID))
        data = json.load(file)
        file.close()
        print(ID + " open")
        for sentence in data :
            for frame in sentence["frames"] :
                if frame["target"]["name"] in frames_to_keep :
                    df.loc[index,frame["target"]["name"]] += 1
    df.to_pickle("data.pkl")
    


def identical_words(L,M) :   # given two lists of words, computes how many words appear in both lists
    count = 0
    
    if len(M) > len(L) :
        for elt in L :
            if elt in M :
                count += 1
    else :
        for elt in M :
            if elt in L :
                count += 1
                
    return count
                