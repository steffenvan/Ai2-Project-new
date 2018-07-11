from extraction import *
from path import data_path
import os
from build_matrix import *
import pandas as pd
from similarity_utility import *
from frame_similarity import get_document_text, tfidf_vectorize_document
from vect_functions import sentence_vec, cosine_vec

json_path = os.path.join(data_path, "json/")

df = load_dataframe()
example_frame = "Cause_to_make_progress"
ref = df.index[8]
candidates = most_similar(ref, 50, df)


sentences_ref = []
doc_text                = get_document_text(ref)
weighted_document       = tfidf_vectorize_document(doc_text, 20)
important_words = weighted_document.keys()
doc_frames = extract_frame_sentence(os.path.join(json_path, candidate), df)
for sentence in doc_frames[example_frame] :
        if sum([word.lower() in sentence.lower() for word in important_words]) > 6 :
            sentences_ref.append(sentence)

docs = []

for candidate in candidates[:6] :
    sentences = []
    doc_text                = get_document_text(candidate)
    weighted_document       = tfidf_vectorize_document(doc_text, 20)
    important_words = weighted_document.keys()
    doc_frames = extract_frame_sentence(os.path.join(json_path, candidate), df)
    # print(doc_frames)
    # for frame in doc_frames.keys() :
    #     for sentence in doc_frames[frame] :
    #             if sum([word.lower() in sentence.lower() for word in important_words]) > 6 :
    #                 print(sentence)
    #                 print("\n")
    for sentence in doc_frames[example_frame] :
            if sum([word.lower() in sentence.lower() for word in important_words]) > 6 :
                sentences.append(sentence)
                print(sentence)
                print("\n")
    docs.append(sentences)

max = 0

for doc in docs :
    for sentence in doc :
        vect_doc = 
        for sentence_ref in sentences_ref :
            
    
    

