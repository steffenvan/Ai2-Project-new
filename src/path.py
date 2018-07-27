import os
import sys
from pathlib import Path

def correct_path(current_directory):
    current_directory = str(current_directory)
    if "src/preprocessing" in current_directory:
        current_directory = Path(current_directory).parents[1]

    elif "src/frames" in current_directory:
        current_directory = Path(current_directory).parents[1]

    elif "src/visualization" in current_directory:
        current_directory = Path(current_directory).parents[1]

    else:
        current_directory = Path(current_directory).parent

    return current_directory

# Setting the correct paths to the parent folders of the project repository.
real_root        = correct_path(Path.cwd())
parent           = Path.resolve(real_root.joinpath(real_root.parent))
data_path        = parent.joinpath("data/")
run_semafor_path = parent.joinpath("semafor/bin/runSemafor.sh")

# print("Root folder:            ", real_root)
# print("Parent of project path: ", parent)
# print("Semafor path:           ", run_semafor_path)

# train data
train_path      = data_path.joinpath("train/")
json_train_path = train_path.joinpath("json/")
txt_train_path  = train_path.joinpath("txt/")
path_to_model = train_path.joinpath("myfullmodel.gsm")
# test data
test_path      = data_path.joinpath("test/")

json_test_path = test_path.joinpath("json/")
txt_test_path  = test_path.joinpath("txt/")
xml_test_path  = test_path.joinpath("xml/")
rej_test_path  = test_path.joinpath("rejected_xml/")

# print(json_test_path)
# print(txt_test_path)
# print(xml_test_path)
# print(rej_test_path)

# the following function can take filenames (with or without paths) or just doc_id as input, and gives the absolute path to the txt or json file as an output

def get_path(doc_id, json = True, train = True) :
    doc_id = os.path.basename(doc_id)
    if doc_id.endswith(".txt") :
        doc_id = doc_id[:-4]
    if doc_id.endswith(".json") :
        doc_id = doc_id[:-5]
    if json :
        filename = doc_id + ".json"
        if train :
            path = os.path.join(json_train_path, filename)
        else :
            path = os.path.join(json_test_path, filename)
    else :
        filename = doc_id + ".txt"
        if train :
            path = os.path.join(txt_train_path, filename)
        else : 
            path = os.path.join(txt_test_path, filename)
    return Path(path)
