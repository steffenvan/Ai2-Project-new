from article import *
import os
from preprocess_functions import *
import sys
from pathlib import Path
# Creating path to the path.py file
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *

# print("\npipeline file:\n")
# print("current is:", Path.cwd())
# print("data folder is:", src)

"""
to add new xml files to the database, simply drag them to data/xml/ and then run this script
"""


corpus_words_path = data_path.joinpath("corpus_words.txt")

assert Path.is_file(corpus_words_path), "please add corpus_words.txt in the data folder"

L = []

for filename in os.listdir(xml_test_path):
    if filename.endswith(".xml") :
        print(filename)
        L.append(Article(filename[:-4]))

print("Currenlty " + str(len(L)) + " files fully pre-processed." )
