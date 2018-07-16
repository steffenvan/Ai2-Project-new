from pathlib import Path
import sys 
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
import json
import string
import os
import pandas as pd


"""
This file allows the extracion of only specific sections of json / txt files
"""

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

def extract_frame_sentence(json_filename, df = pd.DataFrame()):   # if df == "NONE" : the dataframe is loaded from its orginal location (in the data folder)
    if len(df) == 0 :                                      # else (better), specify an already loaded df
        df = load_dataframe()
    json_object = json.load(open(json_filename))
    sentence = ""
    frame_and_sentence = {}
    for list_of_frames in json_object :
        for frame in list_of_frames["frames"]:
            if frame["target"]["name"] in df.columns:
                sentence = " ".join(list_of_frames["tokens"])
                key = frame["target"]["name"]
                if key not in frame_and_sentence :
                    frame_and_sentence.update({key : [sentence]})
                else :
                    if sentence not in frame_and_sentence[key] :
                        frame_and_sentence[key].append(sentence)
    return frame_and_sentence
