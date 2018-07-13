import os
from preprocess_functions import *
import sys
from pathlib import Path
curr_file = (os.path.join(os.getcwd(), sys.argv[0]))
(sys.path.append(str(Path(curr_file).parent.parent)))

from path import *
class Article :

    def __init__(self, filename) :
        xml_extension = filename + ".xml"
        txt_extension = filename + ".txt"
        json_extension = filename + ".json"
        print(os.path.join(xml_test_path, xml_extension))

        assert(os.path.exists(os.path.join(xml_test_path, xml_extension))) , "No matching xml file has been found"
        self.xml = os.path.join(xml_test_path, xml_extension) 
        self.txt = os.path.join(txt_test_path, txt_extension) 
        self.json = os.path.join(json_test_path, json_extension) 

        print("now processing " + filename)

        if (os.path.exists(self.txt) == 0) :
            try :
                print("Creating text file from xml...")
                print(data_path)
                preprocess_file(self.xml, self.txt, os.path.join(data_path, "dico.txt"), os.path.join(data_path, "corpus_words.txt"), 2)
                print("Text file created.")
                print(data_path)
                augment_word_list(self.txt, os.path.join(data_path, "corpus_words.txt"), os.path.join(data_path, "dico.txt"))
            except :
                command = "mv " + self.json + " " + os.path.join(rej_test_path, xml_extension)
                os.system(command)
                print("xml file rejected")

        else :
            print("Text file already exists.")

        if (os.path.exists(self.json) == 0) :
            command = str(run_semafor_path) + " " + self.txt + " " + self.json + " 4"
            os.system(command)
            print("Json file created.")
            

        else :
            print("Json file already exists.")
