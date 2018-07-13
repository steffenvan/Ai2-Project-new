from extraction import *
from path import data_path
import os
from build_matrix import *
import pandas as pd
from similarity_utility import *
from frame_similarity import get_document_text, tfidf_vectorize_document
from vect_functions import sentence_vectorize, cosine_sim
import time

json_path = os.path.join(data_path, "json/")
txt_path = os.path.join(data_path, "txt/")

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']

df = load_dataframe()
# example_frame = "Cause_to_make_progress"
# ref_doc_id = df.index[970]
# candidates = most_similar(ref_doc_id, 50, df)
# ref_frame_sentences = extract_frame_sentence(os.path.join(json_path, ref_doc_id), df)

# Returns the top n highest tfidf valued words as a list.
def important_tfidf_words(doc_id):
    doc_text                      = get_document_text(doc_id)
    tfidf_important_document       = tfidf_vectorize_document(doc_text, 10)
    important_document_words       = tfidf_important_document.keys()
    return important_document_words

# Returns the related sentences containing the specified frame, to the highest tfidf valued words in the doc
def important_sents_of_frame(doc_id, frame_sentences, important_words, min_important_words = 4):
    important_frame_sentences = {}
    # print(frame_sentences)
    for frame_name in frame_sentences.keys() :
        important_sentences = []
        for sentence in frame_sentences[frame_name] :
            if sum([word.lower() in sentence.lower() for word in important_words]) > min_important_words :   # if the number of "important words" in a sentence is more than n, we keep that sentence
                important_sentences.append(sentence)
            important_frame_sentences.update({frame_name:important_sentences})
    # print(important_frame_sentences)
    return important_frame_sentences


def compare(doc1, doc1_important_frame_sentences, important_words1, doc2, doc2_important_frame_sentences, important_words2) :
    score = 0
    i = 0
    for frame_name in frames_to_keep :
        if frame_name in doc1_important_frame_sentences.keys() and frame_name in doc2_important_frame_sentences.keys() :
            score_frame = 0
            sentences1 = doc1_important_frame_sentences[frame_name]
            sentences2 = doc2_important_frame_sentences[frame_name]
            # print(sentences1)
            # print(sentences2)
            for sent1 in sentences1 :
                L = sentence_vectorize(sent1, important_words1)
                for sent2 in sentences2 :
                    M = sentence_vectorize(sent2, important_words2)
                    score_frame += cosine_sim(L,M)
            if len(doc1_important_frame_sentences[frame_name]) > 0 and len(doc2_important_frame_sentences[frame_name]) > 0:        
                score += score_frame/(len(sentences1)*len(sentences2))
                i += 1
    try :
        score /= i
    except :
        return 0
    return score
        


doc_1 = df.index[6729]     # 6729     8940
# doc_2 = df.index[10]

d1_fs = extract_frame_sentence(os.path.join(json_path, doc_1), df)
# d2_fs = extract_frame_sentence(os.path.join(json_path, doc_2), df)

d1_iw = important_tfidf_words(doc_1)
# d2_iw = important_tfidf_words(doc_2)
d1_ifs = important_sents_of_frame(doc_1, d1_fs, d1_iw)
# d2_ifs = important_sents_of_frame(doc_2, d2_fs, d2_iw)

candidates = most_similar(doc_1, 250, df)

scores = []

max_score = 0
best_sugg = ""

for doc_2 in candidates :
    print("ref : ", doc_1, ", current : ", doc_2)
    d2_fs = extract_frame_sentence(os.path.join(json_path, doc_2), df)
    d2_iw = important_tfidf_words(doc_2)
    d2_ifs = important_sents_of_frame(doc_2, d2_fs, d2_iw)
    score = compare(doc_1, d1_ifs, d1_iw, doc_2, d2_ifs, d2_iw)
    if score > max_score :
        max_score = score
        best_sugg = doc_2
    scores.append(score)
    print("***********************")
    print("\n")

os.system("open " + os.path.join(txt_path, doc_1[:-4] + "txt") + " " +  os.path.join(txt_path, best_sugg[:-4] + "txt"))


# Highest tfidf-valued sentences the given frames of the reference file.
# ref_important_words = important_tfidf_words(ref_doc_id)
# sents_of_ref = important_sents_of_frame(ref_doc_id, ref_frame_sentences, ref_important_words)
# # print(sents_of_ref)
# 
# start = time.time()
# 
# for sent in sents_of_ref:
# 
#     # print("Reference sentences: ", sents_of_ref)
#     ref_sent_vect = sentence_vectorize(sent, ref_important_words)
#     # print(sent_vect)
#     for candidate in candidates :
#         print(candidate)
#         candidate_words      = important_tfidf_words(candidate)
#         cand_frame_sentences = extract_frame_sentence(os.path.join(json_path, candidate), df)
#         if example_frame not in cand_frame_sentences.keys() :
#             print("frame not in candidate")
#         else :
#             important_cand_frame_sentences = important_sents_of_frame(candidate, cand_frame_sentences, candidate_words)
# 
#         # print(candidate)
#         print("\n")
#         # print(important_cand_frame_sentences)
#         print("\n")
#     print("****************************")
# 
#     print("//////////////////////////////////////////////////////////////")
# #
# end = time.time()
# print(str(end-start))
#
#
#
#


# for candidate in candidates[:6] :
#     sentences = []
#     doc_text                = get_document_text(candidate)
#     important_document       = tfidf_vectorize_document(doc_text, 20)
#     important_words = important_document.keys()
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
