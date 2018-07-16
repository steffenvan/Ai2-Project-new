# This computes the similarity of the common frames of an abstract with all
# the other abstracts we have.

from file_content_extraction import *
import operator
from similarity_utility import *
from distance_measures import *

# Computes the similarity value of the full sentence of the common frames of the reference abstract
# and all the other abstracts one by one.

# Dictionary for the paper and its similarity value with the reference paper.
paper_and_cos_val = {}
paper_and_dice_val = {}
paper_and_jaccard_val = {}
paper_and_wm_val = {}
paper_and_embeddings_cos_val = {}

if  __name__ == "__main__" :
    df                     = load_dataframe()
    # reference_file         = json.load(open(json_path + str(df.index[0])))
    all_other_abstracts    = df.index[1:]
    ref_content            = get_document_text(df.index[0])
    reference_frames       = get_frames_count(df.index[0])
    ref_frame_and_sentence = extract_frame_sentence(json_train_path.joinpath(str(df.index[0])))

    for abstract_id in all_other_abstracts:
        current_frame_count = get_frames_count(abstract_id)
        sim_frame_count     = common_frames(reference_frames, current_frame_count)

        total_sim_val       = 0.0
        total_dice_val      = 0.0
        total_jaccard_val   = 0.0
        total_wm_val        = 0.0
        total_embed_val     = 0.0

        if sim_frame_count > 4 :
            print("\n****************************************************************\n")
            print(abstract_id, "\n")

            # temp_abstract           = json.load(open(json_path + str(abstract_id)))
            temp_frame_and_sentence = extract_frame_sentence(json_train_path.joinpath(str(abstract_id)))

            doc_text                = get_document_text(abstract_id)
            weighted_document       = tfidf_vectorize_document(doc_text)

            print(weighted_document)
            print("\n")
            # Find common frames between the reference and other abstracts.
            for key in ref_frame_and_sentence.keys() & temp_frame_and_sentence.keys():
                for sentence in temp_frame_and_sentence[key] :
                    print(sentence)
                # Resetting similarity value for each frame
    #             cos_sim       = 0.0
    #             dice_sim      = 0.0
    #             jaccard_sim   = 0.0
    #             wm_sim        = 0.0
    #             embed_cos_sim = 0.0
    #
    #             # Cosine similarity
    #             cos_sim            = tfidf_vector_similarity(ref_frame_and_sentence[key],
    #                                                          temp_frame_and_sentence[key])
    #             total_sim_val     += cos_sim
    #
    #             # dice similarity
    #             dice_sim           = dice_coefficient(ref_frame_and_sentence[key],
    #                                                   temp_frame_and_sentence[key])
    #             total_dice_val    += dice_sim
    #
    #             # jaccard similarity
    #             jaccard_sim        = jaccard_sim_coefficient(ref_frame_and_sentence[key],
    #                                                          temp_frame_and_sentence[key])
    #             total_jaccard_val += jaccard_sim
    #
    #             # Word mover distance
    #             wm_sim             = WMD(ref_frame_and_sentence[key],
    #                                      temp_frame_and_sentence[key])
    #             total_wm_val      += wm_sim
    #
    #             # cosine similarity between word embedding vectors
    #             embed_cos_sim      = embedding_sentence_similarity(ref_frame_and_sentence[key],
    #                                                                temp_frame_and_sentence[key])
    #             total_embed_val   += embed_cos_sim
    #
    #             paper_and_cos_val.update({abstract_id:total_sim_val})
    #             paper_and_dice_val.update({abstract_id:total_dice_val})
    #             paper_and_jaccard_val.update({abstract_id:total_jaccard_val})
    #             paper_and_wm_val.update({abstract_id:total_wm_val})
    #             paper_and_embeddings_cos_val.update({abstract_id:total_embed_val})
    #
    #             print("Similar frame: %s" % key)
    #             print("Cosine similarity: %f" % cos_sim)
    #             print("Dice coefficient: %f" % dice_sim)
    #             print("Jaccard similarity: %f" % jaccard_sim)
    #             print("Word Mover similarity: %f" % wm_sim)
    #             print("Embeddings cosine similarity: %f" % embed_cos_sim)
    #             print("Reference text: %s" % ref_frame_and_sentence[key])
    #
    #             print("Compared text: %s\n" % temp_frame_and_sentence[key])
    #         print("Total cos value: %f" % total_sim_val)
    #         print("Total dice val: %f" % total_dice_val)
    #         print("Total jaccard val: %f" % total_jaccard_val)
    #         print("Total Word Mover val: %f" % total_wm_val)
    #         print("Total embeddings cos value: %f" % total_embed_val)
    #         #print("Total sentence val: %f" % total_sent_val)
    #
    # #Ranking the papers from highest to lowest
    # sorted_cos          = sorted(paper_and_cos_val.items(), key=lambda x: x[1], reverse=True)
    # sorted_dice         = sorted(paper_and_dice_val.items(), key=lambda x: x[1], reverse=True)
    # sorted_jaccard      = sorted(paper_and_jaccard_val.items(), key=lambda x: x[1], reverse=True)
    # sorted_wm           = sorted(paper_and_wm_val.items(), key=lambda x: x[1], reverse=True)
    # sorted_emb_cos      = sorted(paper_and_embeddings_cos_val.items(), key=lambda x: x[1], reverse=True)
    #
    # best_paper_cos      = max(paper_and_cos_val.items(), key=operator.itemgetter(1))[0]
    # best_paper_dice     = max(paper_and_dice_val.items(), key=operator.itemgetter(1))[0]
    # best_paper_jaccard  = max(paper_and_jaccard_val.items(), key=operator.itemgetter(1))[0]
    # best_paper_wm       = max(paper_and_wm_val.items(), key=operator.itemgetter(1))[0]
    # best_paper_emb_cos  = max(paper_and_embeddings_cos_val.items(), key=operator.itemgetter(1))[0]
    #
    # print("\nPaper and Cos values:\n", sorted_cos)
    # print("\nPaper and Dice values:\n", sorted_dice)
    # print("\nPaper and Jaccard values:\n", sorted_jaccard)
    # print("\nPaper and word mover values:\n", sorted_wm)
    # print("\nPaper and embeddings cosine similarity values:\n", sorted_emb_cos)
    # print("\nTotal similar papers:", len(paper_and_cos_val))
    # print("\nMost similar paper using cosine:", best_paper_cos)
    # print("\nMost similar paper using dice:", best_paper_dice)
    # print("\nMost similar paper using jaccard:", best_paper_jaccard)
    # print("\nMost similar paper using word mover:", best_paper_wm)
    # print("\nMost similar paper using embeddings and cosine similarity:", best_paper_emb_cos)
