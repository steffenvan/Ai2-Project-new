import pandas as pd
import numpy as np
import json
import os


ids = []
json_path = "data/json_abs"
json_filename_endings = "-parscit.130908.json"
frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply']

for filename in os.listdir(json_path):
    ids.append(filename[:8])

df = pd.DataFrame(columns=[["ID"]+frames_to_keep])
df["ID"] = ids
df = df.fillna(value=0)

def matrix_with_ones(df):
    # Loop through all ID's and list present frames in it's json_file. If a frame in frames_to_keep is present, then add a 1.
    for index, ID in df["ID"].iteritems():
        parsing_output = json.load(open(json_path+"/"+ID+json_filename_endings))
        present_frames = []
        for sentence in parsing_output:
            for frame in sentence["frames"] :
                frame_name = frame["target"]["name"]
                if frame_name not in present_frames:
                    present_frames.append(frame_name)
        for frame in frames_to_keep:
            if frame in present_frames:
                df.loc[index,str(frame)] = 1
    return df

matrix = matrix_with_ones(df)
