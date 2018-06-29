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
                    
    df.set_index("ID", inplace = True)
    df.to_pickle("data.pkl")
                