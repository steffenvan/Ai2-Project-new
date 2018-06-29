from path import root
import json
import string


"""
This file allows the extracion of only specific sections of json / txt files
"""

def extract_text(frame, vocab = [], stopwords = [], punct = []) :      # given a frame, extract all the text from frameElements / spans
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
            
    
    
        
    if len(stopwords+punct) :
        tokens = [elt for elt in L if elt not in stopwords+punct+list(" ")]    # remove stopwords / punctuation if specified 
        
    if len(vocab) :
        tokens = [elt for elt in L if elt in vocab]                            # only keep words from a specified vocabulary if specified
        
    # return [tokens_target, tokens_annot]

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