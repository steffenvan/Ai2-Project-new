import pandas as pd
import jsonlines
from collections import defaultdict

file = 'test.jsonl'

# given a jsonlines file, returns a list of dictionariies containing the verb and labels of the SRL parse, for the entire file.
def make_sentence_group_from_file(jsonl_file):
    sentence_groups = []
    labels = ["ARG0","ARG1", "ARG2", "ARG3", "ARG4", "ARGM-TMP", "ARGM-LOC", "ARGM-DIR", "ARGM-MNR", "ARGM-ADV", "ARGM-PRD"]
    with jsonlines.open(jsonl_file) as reader:
        for obj in reader:
            for verb in obj["verbs"]:
                temp_dict = defaultdict(list)
                temp_dict["V"] = verb["verb"]
                for index, tag in enumerate(verb["tags"]):
                    for label in labels:
                        if label in tag:
                            temp_dict[label].append(obj["words"][index])
                sentence_groups.append(temp_dict)
                # remove these print statements later
                print(temp_dict)
                print()
    return sentence_groups

#example
sentence_groups = make_sentence_group_from_file(file)
first_dict = sentence_groups[0]
# The sentences for each label are in lists. To get the joined sentence of one argument:
print(" ".join(first_dict["ARG1"]))


# given an jsonlines object (a sentence/parse of a sentence), returns the groups of that object,
# which can then be used to get each sentence seperately and insert the groups into the desired structure.
def make_sentence_group_from_obj(obj):
    sentence_groups = []
    labels = ["ARG0","ARG1", "ARG2", "ARG3", "ARG4", "ARGM-TMP", "ARGM-LOC", "ARGM-DIR", "ARGM-MNR", "ARGM-ADV", "ARGM-PRD"]
    for verb in obj["verbs"]:
        temp_dict = defaultdict(list)
        temp_dict["V"] = verb["verb"]
        for index, tag in enumerate(verb["tags"]):
            for label in labels:
                if label in tag:
                    temp_dict[label].append(obj["words"][index])
        sentence_groups.append(temp_dict)
        # remove these print statements later
        print(temp_dict)
        print()
    return sentence_groups

#example
with jsonlines.open(file) as reader:
        for obj in reader:
            sentence_groupss = make_sentence_group_from_obj(obj)
            # do something: fx put into our other structure for the frame semantic parsing
