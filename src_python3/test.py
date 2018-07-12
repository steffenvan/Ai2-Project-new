from extraction import *
from path import data_path
import os
from build_matrix import *
import pandas as pd
from similarity_utility import *
from frame_similarity import get_document_text, tfidf_vectorize_document
from vect_functions import sentence_vectorize, cosine_sim

json_path = os.path.join(data_path, "json/")

df = load_dataframe()
example_frame = "Cause_to_make_progress"
ref_doc_id = df.index[9]
candidates = most_similar(ref_doc_id, 50, df)
ref_sent_frames = extract_frame_sentence(os.path.join(json_path, ref_doc_id), df)

# Returns the frame names of the top n highest tfidf valued words.
def weighted_tfidf_words(doc_id):
    doc_text                      = get_document_text(doc_id)
    tfidf_weighted_document       = tfidf_vectorize_document(doc_text, 20)
    weighted_document_words       = tfidf_weighted_document.keys()
    return weighted_document_words

# Returns the related sentences to the highest tfidf valued words in the reference file
def weighted_sents_of_frame(doc_id, frame_sentences, frame, weighted_words):
    all_weighted_sentences = []
    for sentence in frame_sentences[frame] :
        if sum([word.lower() in sentence.lower() for word in weighted_words]) > 2 :   # if the number of "important words" in a sentence is more than n, we keep that sentence
            all_weighted_sentences.append(sentence)

    return all_weighted_sentences

ref_weighted_words = weighted_tfidf_words(ref_doc_id)

# def compare(doc1, doc2) :
#     for frame_name in relevant_frames :
#         for sent1 in sents_of_frame(doc1, frame_name) :
#             L =
#             for sent2 in sents_of_frame(doc2, frame_name) :
#                 score_frame += cos


# Highest tfidf-valued sentences the given frames of the reference file.
ref_weighted_words = weighted_tfidf_words(ref_doc_id)
sents_of_ref = weighted_sents_of_frame(ref_doc_id, ref_sent_frames, example_frame, ref_weighted_words)
print(sents_of_ref)

for sent in sents_of_ref:
    # print("Reference sentences: ", sents_of_ref)
    ref_sent_vect = sentence_vectorize(sent, ref_weighted_words)
    # print(sent_vect)
    for candidate in candidates[:6] :
        candidate_words      = weighted_tfidf_words(candidate)
        # weighted_candidate_sentences = weighted_sents
        cand_frame_sentences = extract_frame_sentence(os.path.join(json_path, candidate), df)
        weighted_cand_frame_sentences = weighted_sents_of_frame(candidate, cand_frame_sentences, example_frame, candidate_words)
        print(candidate)
        print("\n")
        print(weighted_cand_frame_sentences)
        print("\n")
    print("****************************")

    print("//////////////////////////////////////////////////////////////")
#
#
#
#
#
#
#


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
