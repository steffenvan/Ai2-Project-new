from path import *
from extraction import *
import os
import pandas as pd
import spacy
import json
import pickle

"""
This script allows the creation of two files :
- a data frame, stored as data.pkl, containing for each files the number of occurences of the frames specified in the frames_to_keep list specified below
- a list of dictionnaries, stored as "out.pkl". Each entry of the list is a dictionnary, with a  

"""

json_path = os.path.join(data_path + "json/")


frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply', 'Accomplishment']

def build_matrix() :
    i = 0
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)
    ids = [filename for filename in os.listdir(json_path) if filename.endswith("json")]
    total = len(ids)
    df["ID"] = ids
    frames_text = []
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        try :
            print("opening " + ID)
            print(i, " on ", total)
            file = open(str(json_path+"/"+ID))
            data = json.load(file)
            file.close()
            d = {} ##
            print(ID + " open")
            for sentence in data :
                for frame in sentence["frames"] :
                    if frame["target"]["name"] in frames_to_keep :
                        df.loc[index,frame["target"]["name"]] += 1
                        if frame["target"]["name"] not in d :
                            d[frame["target"]["name"]] = []
                            d[frame["target"]["name"]].append(extract_text(frame))
            frames_text.append(d)
        except :
            print(ID + " could not be opened")                
        i += 1
    df.set_index("ID", inplace = True)
    df.to_pickle(os.path.join(data_path,"data.pkl"))

    output_file = open(os.path.join(data_path,"frames_text.pkl"),"wb+")
    pickle.dump(frames_text, output_file)
    output_file.close()
    
    
def build_abs_matrix() :                    # doesn't work for now because of abstract_json()
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)
    ids = [filename for filename in os.listdir(json_path) if filename.endswith("json")]
    df["ID"] = ids
    frames_text = []
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        # try :
        print("opening " + ID)
        # file = open(str(json_path+"/"+ID))
        # data = json.load(file)
        data = abstract_json(str(json_path+"/"+ID))
        # file.close()
        d = {} ##
        print(ID + " open")
        for sentence in data :
            for frame in sentence["frames"] :
                if frame["target"]["name"] in frames_to_keep :
                    df.loc[index,frame["target"]["name"]] += 1
                    if frame["target"]["name"] not in d :
                        d[frame["target"]["name"]] = []
                        d[frame["target"]["name"]].append(extract_text(frame))
        frames_text.append(d)
        # except :
        #     print(ID + " could not be opened")                

    df.set_index("ID", inplace = True)
    df.to_pickle(os.path.join(data_path,"data_abs.pkl"))

    output_file = open(os.path.join(data_path,"frames_text_abs.pkl"),"wb+")
    pickle.dump(frames_text, output_file)
    output_file.close()