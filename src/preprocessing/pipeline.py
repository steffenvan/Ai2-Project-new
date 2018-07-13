from article import *
import os
from preprocess_functions import *
import sys
from pathlib import Path
curr_file = (os.path.join(os.getcwd(), sys.argv[0]))
src_path = Path(curr_file).parent.parent
(sys.path.append(str(Path(curr_file).parent.parent)))

from path import *

src = correct_path(root, str(Path.cwd()))
print(root)
print("current is:", Path.cwd())
print("data folder is:", src)
# if "src/preprocessing" or "src/visualization" in Path.cwd():
#     root = Path.cwd().parent.parent
# else:
#     root = Path.cwd().parent

"""
to add new xml files to the database, simply drag them to data/xml/ and then run this script
"""
print("data path is:", data_path)

assert os.path.isfile(os.path.join(data_path, "corpus_words.txt")), "please add corpus_words.txt in the data folder"

L = []

for filename in os.listdir(xml_test_path) :
    if filename.endswith(".xml") :
        print(filename)
        L.append(Article(filename[:-4]))

print("Currenlty " + str(len(L)) + " files fully pre-processed." )