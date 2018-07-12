from extraction import *
from path import data_path
import os
from build_matrix import *
import pandas as pd
from similarity_utility import *
from frame_similarity import get_document_text, tfidf_vectorize_document
from vect_functions import sentence_vec, cosine_sim

json_path = os.path.join(data_path, "json/")

df = load_dataframe()
example_frame = "Cause_to_make_progress"
ref = df.index[9]
candidates = most_similar(ref, 50, df)


sentences_ref = []
doc_text                = get_document_text(ref)
weighted_document       = tfidf_vectorize_document(doc_text, 50)
important_words_ref = weighted_document.keys()
doc_frames = extract_frame_sentence(os.path.join(json_path, ref), df)
for sentence in doc_frames[example_frame] :
        if sum([word.lower() in sentence.lower() for word in important_words_ref]) > 2 :   # if the number of "important words" in a sentence is more than n, we keep that sentence
            sentences_ref.append(sentence)

docs = []

for sentence_ref in sentences_ref :
    print("reference : ", sentence_ref)
    L = sentence_vec(sentence_ref, important_words_ref)
    for candidate in candidates[:6] :
        doc_text                = get_document_text(candidate)
        weighted_document       = tfidf_vectorize_document(doc_text, 50)
        important_words = weighted_document.keys()
        doc_frames = extract_frame_sentence(os.path.join(json_path, candidate), df)
        for sentence in doc_frames[example_frame] :
            if sum([word.lower() in sentence.lower() for word in important_words]) > 2 :
                M =  sentence_vec(sentence, important_words)
                if cosine_sim(L,M) > 0.8 :
                    print("***", sentence, cosine_sim(L,M))
        print("****************************")
    
    print("//////////////////////////////////////////////////////////////")





sents_of_frame(doc, frame, doc_frame)

def compare(doc1, doc2) :
    for frame_name in relevant_frames :
        for sent1 in sents_of_frame(doc1, frame_name) :
            L = 
            for sent2 in sents_of_frame(doc2, frame_name) :
                score_frame += cos





# for candidate in candidates[:6] :
#     sentences = []
#     doc_text                = get_document_text(candidate)
#     weighted_document       = tfidf_vectorize_document(doc_text, 20)
#     important_words = weighted_document.keys()
#     doc_frames = extract_frame_sentence(os.path.join(json_path, candidate), df)
#     for sentence in doc_frames[example_frame] :
#         print(sentence)
#         L = sentence_vec(sentence, important_words)
#         if sum([word.lower() in sentence.lower() for word in important_words]) > 6 :
#             for sentence_ref in sentences ref :
#                 M =  sentence_vec(sentence_ref, important_words_ref)
#             # sentences.append(sentence)
#                 print("***", sentence)
#     print("****************************")
#     docs.append(sentences)
# 
# 
# max = 0
# 
# for doc in docs :
#     for sentence in doc :
#         vect_doc = 
#         for sentence_ref in sentences_ref :
            
    
    

