from path import *
import os
import json
from extraction import *

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']
# json_abs = parent + "/json_abs/"
# 
# counts = {}
# 
# for file in os.listdir(json_abs) :
#     if file.endswith(".json") :
#         data = json.load(open(json_abs+file))
#         for sentence in data :
#             for frame in sentence["frames"] :
#                 name = frame["target"]["name"]
#                 if name in counts :
#                     counts[name] += 1
#                 else :
#                     counts[name] = 1
# 
# tmp = []
# 
# # iterate through the dictionary and append each tuple into the temporary list 
# for key, value in counts.items():
#     tmptuple = (value, key)
#     tmp.append(tmptuple)
# 
# # sort the list in ascending order
# tmp = sorted(tmp, reverse = True)
# 
# print (tmp)

files = os.listdir(abs_path)[:100]
# path = parent
# print(path)
for file in files :
# file = os.listdir(abs_path)[3]
    print(file)
# print(path_to_file)
    data = json.load(open(abs_path + file))

    for sentence in data :
        for frame in sentence["frames"] :
            if frame["target"]["name"] in frames_to_keep :
                print("*******************")
                print(extract_with_pos(frame))
            # print(frame["target"]["name"])
            # print(frame["target"]["spans"][0]["text"])
            # if len(frame["annotationSets"][0]["frameElements"]) :
            #     # print(frame["annotationSets"][0]["frameElements"][0]["spans"][0]["text"])
            #     for elt in frame["annotationSets"][0]["frameElements"] :
            #         print(elt["spans"][0]["text"])


