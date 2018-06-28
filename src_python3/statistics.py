from path import *
import os
import json
from extraction import *

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

files = os.listdir(abs_path)[2:4]
# path = parent
# print(path)
file = os.listdir(abs_path)[3]
print(file)
# print(path_to_file)

data = json.load(open(abs_path + file))

for sentence in data :
    for frame in sentence["frames"] :
        print("*******************")
        print(frame["target"]["spans"][0]["text"])
        if len(frame["annotationSets"][0]["frameElements"]) :
            print(frame["annotationSets"][0]["frameElements"][0]["spans"][0]["text"])
