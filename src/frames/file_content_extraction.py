from pathlib import Path
import sys
import os
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
import json
import string
import pandas as pd

"""
This file allows the extracion of only specific sections of json / txt files
"""

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']

def load_dataframe(file = os.path.join(data_path,"data.pkl")) :
    df = pd.read_pickle(file)
    return df

def inside_json_content(frame) :
    list_of_element = []
    if len(frame["annotationSets"][0]["frameElements"]) :
        for element in frame["annotationSets"][0]["frameElements"] :
            list_of_element.append(element)
    return list_of_element

def extract_text(frame) :      # given a frame, extract all the text from frameElements / spans
    tokens_annot = []
    tokens_name = []
    tokens_target = []
    output  = {}
    tokens_target = list(map(str.lower, frame["target"]["spans"][0]["text"].split()))
    output["target"] = tokens_target
    unwanted_info = ["<","newsection", ">","abstract"]
    list_of_element = inside_json_content(frame)

    for element in list_of_element:
        for word in element["spans"][0]["text"].split(" "):
            if word.lower() not in unwanted_info:
                tokens_annot.append(word.lower())

    output["annot"] = tokens_annot
    output["name"] = frame["target"]["name"]
    
    return output

def extract_with_pos(frame, nlp, to_keep = ["ADJ","NOUN"]) :

        tokens_annot = []
        tokens_name = []
        tokens_target = []
        output  = {}
        text_target = nlp(frame["target"]["spans"][0]["text"])
        for token in text_target :
            if token.pos_ in to_keep :
                tokens_target.append(token.text.lower())
        output["target"] = tokens_target

        if len(frame["annotationSets"][0]["frameElements"]) :
            for elt in frame["annotationSets"][0]["frameElements"] :
                # print(elt["spans"][0]["text"])
                text_annot = nlp(elt["spans"][0]["text"])
                for token in text_annot :
                    if token.pos_ in to_keep :
                        tokens_annot.append(token.text.lower())


        output["annot"] = tokens_annot

        output["name"] = frame["target"]["name"]

        return output


def abstract_json(json_file) :                    # returns the part of the json semafor output correspounding to the abstract
    file = open(json_file)
    full_output = json.load(file)
    file.close()
    i = 0
    while ("Abstract" not in full_output[i]["tokens"]) : # or "abstract" not in full_output[i]["tokens"]
        i += 1
    beg = i
    # introduction_trigg = ["Introduction","introduction", ]
    while ("newSection" not in full_output[i]["tokens"]) :
        i += 1
    end = i
    return full_output[beg:end]


def conclusion_json(json_file) :                  # file = path / filename
        full_output = json.load(open(json_file))
        i = 0
        while ("Conclusion" not in full_output[i]["tokens"] and "newSection" not in full_output[i]["tokens"]) :
            i += 1
        beg = i
        i += 1
        while ("newSection" not in full_output[i]["tokens"]) :
            i += 1
        end = i
        return full_output[beg:end]

def abstract_txt(txt_file) :
    full = open(txt_file).read().split('\n')
    i = 0
    while (("Abstract" not in full[i]) or ("<newSection>" not in full[i])) :
        i += 1
    beg = i
    i += 1
    while ("<newSection>" not in full[i]) :
        i += 1
    end = i
    return '\n'.join(full[beg:end])

def conclusion_txt(txt_file) :
        full = open(txt_file).read().split('\n')
        i = 0
        while (("Conclusion" not in full[i]) or ("<newSection>" not in full[i])) :
            i += 1
        beg = i
        i += 1
        while ("<newSection>" not in full[i]) :
            i += 1
        end = i
        return '\n'.join(full[beg:end])

# Returns a dictionnary with the frame names as keys and a list of sentences containing occurences of that frame as values

def sents_by_frame(doc_id, train):   # if df == "NONE" : the dataframe is loaded from its orginal 
    json_object = json.load(open(get_path(doc_id, True, train)))
    sentence = ""
    output = {}
    for list_of_frames in json_object :
        for frame in list_of_frames["frames"]:
            # if frame["target"]["name"] in df.columns:
            if frame["target"]["name"] in frames_to_keep :
                sentence = " ".join(list_of_frames["tokens"])
                key = frame["target"]["name"]
                if key not in output :
                    output.update({key : [sentence]})
                else :
                    if sentence not in output[key] :
                        output[key].append(sentence)
    return output
    
def important_sents_by_frame(doc_id, train, n_max = 5) :
    frame_sentences = sents_by_frame(doc_id, train)
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