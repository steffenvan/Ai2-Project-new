import os
from pathlib import Path
import sys
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from random import shuffle
from file_content_extraction import *
import sys
sys.path.append(str(Path(curr_file).parents[1]))
from path import *


def display_frame_instances(frame_name, nmax = 5) :    # prints all the instances of a given frame among the toy dataset

    print("**************************************\n")

    files = os.listdir(json_train_path)

    shuffle(files)

    i = 1

    for filename in files :

        if i > nmax :

            break

        if filename.endswith(".json") :


            if i > nmax :

                break



            output = json.load(open(os.path.join(json_train_path, filename)))

            for sentence in output :

                        if i > nmax :

                            break

                        for frame in sentence["frames"] :

                            if i > nmax :

                                break

                            if frame_name in frame["target"]["name"].lower() :

                                # print(frame_name + ": " + " ".join(sentence["tokens"]))
                                print(frame_name + ": " + str(frame))
                                print("\n")

                                i += 1


relevant_frames = ["predicament"] # "accomp","accura","compar","relevant", "competition", "desirability", "scale"

# for frame in relevant_frames :
#     display_frame_instances(frame, 20)

    
def display_frame_elements(frame_name, nmax = 5) :
    print("**************************************\n")

    files = os.listdir(json_train_path)

    shuffle(files)

    i = 1

    for filename in files :

        if i > nmax :

            break

        if filename.endswith(".json") :


            if i > nmax :

                break



            output = json.load(open(os.path.join(json_train_path, filename)))

            for sentence in output :

                        if i > nmax :

                            break

                        for frame in sentence["frames"] :

                            if i > nmax :

                                break

                            if frame_name in frame["target"]["name"].lower() :

                                # print(frame_name + ": " + " ".join(sentence["tokens"]))
                                # print(frame_name + ": " + str(frame))
                                # print("\n")
                                
                                print(str(frame["annotationSets"][0]["frameElements"]))
                                print(frame_name + ": " + " ".join(sentence["tokens"]))
                                print("\n")
                                i += 1
        
for frame in relevant_frames :
    display_frame_elements(frame, 20)
