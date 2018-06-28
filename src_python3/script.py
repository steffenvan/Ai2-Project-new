from path import root, parent
import os
import json

json_abs = parent + "/json_abs/"

for file in os.listdir(json_abs) :
    if file.endswith(".json") :
        data = json.load(open(json_abs+file))
        for sentence in data :
            for frame in sentence["frames"] :
                print(frame)
        

