import os
import sys
from pathlib import Path

def correct_path(current_directory):
    current_directory = str(current_directory)
    if ("src/preprocessing" or "src/frames" or "src/visualization") in current_directory:
        current_directory = Path(current_directory).parents[1]
    else:
        current_directory = Path(current_directory).parent
    return current_directory

# Setting the correct paths to the parent folders of the project repository.
real_root        = correct_path(Path.cwd())
parent           = Path.resolve(real_root.joinpath(real_root.parent))
data_path        = parent.joinpath("data/")
run_semafor_path = parent.joinpath("semafor/bin/runSemafor.sh")

print("Root folder:            ", real_root)
print("Parent of project path: ", parent)
print("Semafor path:           ", run_semafor_path)

# train data

train_path      = data_path.joinpath(data_path, "train/")

json_train_path = train_path.joinpath(train_path, "json/")
txt_train_path  = train_path.joinpath(train_path, "txt/")

# test data
test_path      = data_path.joinpath("test/")

json_test_path = test_path.joinpath("json/")
txt_test_path  = test_path.joinpath("txt/")
xml_test_path  = test_path.joinpath("xml/")
rej_test_path  = test_path.joinpath("rejected_xml/")

print(json_test_path)
print(txt_test_path)
print(xml_test_path)
print(rej_test_path)
