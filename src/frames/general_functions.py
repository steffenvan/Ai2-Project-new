from pathlib import Path
import sys
import os
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *


# doc_id = filename with no extension nor dot

# def get_path(doc_id, json = True, train = True) :
#     if json :
#         filename = doc_id + ".json"
#         if train :
#             path = os.path.join(json_train_path, filename)
#         else :
#             path = os.path.join(json_test_path, filename)
#     else :
#         filename = doc_id + ".txt"
#         if train :
#             path = os.path.join(txt_train_path, filename)
#         else : 
#             path = os.path.join(txt_test_path, filename)
#     return path

print(open(get_path("D15-1260-parscit.130908")).read())