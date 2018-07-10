from path import root, parent, data_path
import json
import string
import spacy
import os
import pandas as pd


"""
This file allows the extracion of only specific sections of json / txt files
"""

# If needed install the core_web package by:
# python3 -m spacy download en_core_web_lg

try :
    nlp = spacy.load("en_core_web_lg")
except :
    os.system("python3 -m spacy download en_core_web_lg")
    nlp = spacy.load("en_core_web_lg")

def load_dataframe(file = os.path.join(data_path,"data.pkl")) :
    df = pd.read_pickle(file)
    return df
    

def extract_text(frame) :      # given a frame, extract all the text from frameElements / spans
    tokens_annot = []
    tokens_name = []
    tokens_target = []
    output  = {}
    tokens_target = list(map(str.lower,frame["target"]["spans"][0]["text"].split()))
    output["target"] = tokens_target

    if len(frame["annotationSets"][0]["frameElements"]) :
        for elt in frame["annotationSets"][0]["frameElements"] :
            # print(elt["spans"][0]["text"])
            for word in elt["spans"][0]["text"].split(" ") :
                if word.lower() not in ["<","newsection", ">","abstract"] :
                    tokens_annot.append(word.lower())

    output["annot"] = tokens_annot

    output["name"] = frame["target"]["name"]

    return output


def extract_with_pos(frame, to_keep = ["ADJ","NOUN"]) :

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
    full_output = json.load(open(json_file))
    i = 0
    while ("Abstract" not in full_output[i]["tokens"]) : # or "abstract" not in full_output[i]["tokens"]
        i += 1
    beg = i
    while ("Introduction" not in full_output[i]["tokens"]) :
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
        
def extract_frame_sentence(json_filename, df = "NONE"):   # if df == "NONE" : the dataframe is loaded from its orginal location (in the data folder)
    if df == "NONE" :                                      # else (better), specify an already loaded df
        df = load_dataframe()
    json_object = json.load(open(json_filename))
    sentence = ""
    frame_and_sentence = {}
    for list_of_frames in json_object :
        for frame in list_of_frames["frames"]:
            if frame["target"]["name"] in df.columns:
                sentence = " ".join(list_of_frames["tokens"])
                frame_and_sentence.update({frame["target"]["name"] : sentence})

    return frame_and_sentence
