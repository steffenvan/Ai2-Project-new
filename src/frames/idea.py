import json
from file_content_extraction import *
from path import data_path, path_to_model
import os
# from build_matrix import *
import pandas as pd
from similarity_utility import *
from distance_measures import *
import time
from gensim.models import Word2Vec
json_path = os.path.join(data_path, "train/json/")
txt_path = os.path.join(data_path, "train/txt/")
from random import shuffle

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']

frames_with_fe = ["Means", "Assessing", "Usefulness", "Reasoning", "Cause_to_make_progress", "Importance", "Desirability", "Evaluative_comparison","Trust", "Position_on_a_scale", "Predicament", "Supply"]

def sents_by_frame_with_fe(doc_id, train): 
    json_object = json.load(open(get_path(doc_id, True, train)))
    sentence = ""
    output = {}
    for list_of_frames in json_object :
        for frame in list_of_frames["frames"]:
            # if frame["target"]["name"] in df.columns:
            if frame["target"]["name"] in frames_with_fe :
                sentence = " ".join(list_of_frames["tokens"])
                key = frame["target"]["name"]
                if key not in output :
                    output.update({key : [frame]})
                else :
                    if sentence not in output[key] :
                        output[key].append(sentence)
    return output
    
def important_sents_by_frame_with_fe(doc_id, train, n_max = 5) :
    frame_sentences = sents_by_frame_with_fe(doc_id, train)
    output = {}
    for frame_name in frame_sentences.keys() :
        temp_dict = {}  # maps the tfidf values to each sentence containing "frame_name"
        sentences = frame_sentences[frame_name]
        for sentence in sentences :
            mean = 0
            word_counter = 0
            for word in sentence.split(" ") :
                try :
                    mean += tfidf_value(word.lower(), get_path(doc_id, False, train), map_file, vocabulary, X)
                    word_counter += 1.0
                except :
                    a = 0
            if word_counter == 0 :
                mean = 0
            else :
                mean /= word_counter
            temp_dict.update({mean : sentence})
        L = []       # list of the sentences to keep
        sorted_keys = sorted(list(temp_dict.keys()), reverse = True)
        for key in sorted_keys[:n_max] :
            L.append(temp_dict[key])
        output.update({frame_name : L})
    return output