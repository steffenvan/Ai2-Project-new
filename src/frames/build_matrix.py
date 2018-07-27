from file_content_extraction import *
import spacy
import pickle
from tfidf_functions import highest_tfidf_bigrams, load_tfidf_bigrams
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))

from path import *

"""
This script allows the creation of two files :
- a data frame, stored as data.pkl, containing for each files the number of
- occurences of the frames specified in the frames_to_keep list specified below
- a list of dictionnaries, stored as "out.pkl". Each entry of the list is a dictionnary, with a
"""

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply', 'Accomplishment']


def open_json(filename):
    path_to_json = json_train_path.joinpath(filename)
    with open(path_to_json) as json_file:
        json_data = json.loads(json_file.read())
    return json_data

def is_valid(filename, train):
    json_file = get_path(filename, True, train)
    if json_file.exists():
        try:
            test = open_json(json_file)
            # print(test)
            return test
        except:
            print("Invalid json file: ", filename)
            return False

def create_file_list(path_to_files, extension):
    folder            = os.listdir(path_to_files)
    list_of_filenames = [filename for filename
                        in folder
                        if filename.endswith(extension)]
    return list_of_filenames

def create_pickle(list_of_frames, path, pkl_filename):
    frames_text_path = path.joinpath(pkl_filename)
    output_file      = open(frames_text_path, "wb+")
    result           = pickle.dump(list_of_frames, output_file)
    # print(output_file)
    output_file.close()
    return result

def data_frame_init(important_frames, rows):
    columns          = ["ID"] + important_frames
    data_frame       = pd.DataFrame(columns = columns)
    data_frame.fillna(value = 0, inplace = True)
    data_frame["ID"] = rows
    # print(data_frame["ID"])
    # print(data_frame)
    return data_frame

def update_data_frame(frame, data_frame, index):
    data_frame.loc[index, frame] += 1
    return data_frame



def build_matrix(update = True) :        # if update = True, loads the data_frame from the training data, augments it with the test data. Else, creates a new matrix from the training data
    if update :
        matrix_data_path = json_test_path
    else :
        matrix_data_path = json_train_path
    i = 1
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)
    ids = [filename[:-5] for filename in os.listdir(matrix_data_path) if filename.endswith("json")]     # ids = filenames without extension
    total = len(ids)
    df["ID"] = ids
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        try :
            print("opening " + ID)
            print(i, " on ", total)
            data = is_valid(ID, not update)
            d = {} ##
            print(ID + " open")
            for sentence in data :
                for frame in sentence["frames"] :
                    if frame["target"]["name"] in frames_to_keep :
                        df.loc[index,frame["target"]["name"]] += 1
        except :
            print(ID + " could not be opened")
        i += 1
    df.set_index("ID", inplace = True)
    if update :
        old_df = load_dataframe()
        new_df = pd.concat([old_df, df], verify_integrity = True)
        new_df.to_pickle(os.path.join(data_path,"test_data.pkl"))
    else :
        df.to_pickle(os.path.join(data_path,"train_data.pkl"))




def build_tfidf_matrix(map_file, vocabulary, X, update = True, nmax = 10) :        # if update = True, loads the data_frame from the training data, augments it with the test data. Else, creates a new matrix from the training data
    if update :
        matrix_data_path = json_test_path
    else :
        matrix_data_path = json_train_path
    i = 1
    columns = ["ID"] + [str(j) for j in range(1,nmax)]
    df = pd.DataFrame(columns = columns)
    ids = [filename[:-5] for filename in os.listdir(matrix_data_path) if filename.endswith("json")]     # ids = filenames without extension
    total = len(ids)
    df["ID"] = ids
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        try :
            print("processing ", ID)
            print(i, " on ", total)
            best_bigrams = highest_tfidf_bigrams(ID, map_file, vocabulary, X, nmax)
            
            for j in range(1,nmax+1) :
                df.loc[index,str(j)] = best_bigrams[j-1][0]
        except :
            print(ID + " could not be processed")
        i += 1
    df.set_index("ID", inplace = True)
    if update :
        old_df = load_dataframe()
        new_df = pd.concat([old_df, df], verify_integrity = True)
        new_df.to_pickle(os.path.join(data_path,"test_tfidf.pkl"))
    else :
        df.to_pickle(os.path.join(data_path,"train_tfidf.pkl"))


[vectorizer, X, map_file] = load_tfidf_bigrams()
vocabulary = vectorizer.vocabulary_

build_tfidf_matrix(map_file, vocabulary, X, False, 50)





# build_matrix(update = 0)



# # def build_abs_matrix() :                    # doesn't work for now because of abstract_json()
# #     columns = ["ID"] + frames_to_keep
# #     df = pd.DataFrame(columns = columns)
# #     ids = [filename for filename in os.listdir(json_path) if filename.endswith("json")]
# #     df["ID"] = ids
# #     frames_text = []
# #     df.fillna(value = 0, inplace = True)
# #     for index, ID in df["ID"].iteritems():
# #         # try :
# #         print("opening " + ID)
# #         # file = open(str(json_path+"/"+ID))
# #         # data = json.load(file)
# #         data = abstract_json(str(json_path+"/"+ID))
# #         # file.close()
# #         d = {} ##
# #         print(ID + " open")
# #         for sentence in data :
# #             for frame in sentence["frames"] :
# #                 if frame["target"]["name"] in frames_to_keep :
# #                     df.loc[index,frame["target"]["name"]] += 1
# #                     if frame["target"]["name"] not in d :
# #                         d[frame["target"]["name"]] = []
# #                         d[frame["target"]["name"]].append(extract_text(frame))
# #         frames_text.append(d)
# #         # except :
# #         #     print(ID + " could not be opened")
# #
# #     df.set_index("ID", inplace = True)
# #     df.to_pickle(os.path.join(data_path,"data_abs.pkl"))
# #
# #     output_file = open(os.path.join(data_path,"frames_text_abs.pkl"),"wb+")
# #     pickle.dump(frames_text, output_file)
# #     output_file.close()
