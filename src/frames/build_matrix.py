from file_content_extraction import *
import spacy
import pickle

curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))

"""
This script allows the creation of two files :
- a data frame, stored as data.pkl, containing for each files the number of
- occurences of the frames specified in the frames_to_keep list specified below
- a list of dictionnaries, stored as "out.pkl". Each entry of the list is a dictionnary, with a
"""

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply', 'Accomplishment']

    #     frames.append(frame)
    # return frame

# def get_frame_sentence(frame):
#     file_frame = get_frame(frame)
#     for element in file_frame:


def open_json(filename):
    path_to_json = json_train_path.joinpath(filename)
    with open(path_to_json) as json_file:
        json_data = json.loads(json_file.read())
    return json_data

def is_valid(filename):
    json_file = json_train_path.joinpath(filename)
    if json_file.exists():
        try:
            test = open_json(json_file)
            print(test)
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
#
# def create_frame_content(filename, index):
#     frames_text = []
#     valid_file  = is_valid(filename)
#
#     if valid_file:
#         d           = {} ##
#         print(filename + " open")
#
#         for sentence in valid_file:
#             for frame in sentence["frames"]:
#
#                 frame_name = frame["target"]["name"]
#                 if frame_name in frames_to_keep:
#
#                     if frame_name not in d:
#                         d[frame_name] = []
#                         d[frame_name].append(extract_text(frame))
#         frames_text.append(d)
#     # else:
#     #     os.remove(json_train_path.joinpath(filename))
#
#     return frames_text


def build_matrix():
    i = 0
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)
    ids = [filename for filename in os.listdir(json_train_path) if filename.endswith("json")]
    total = len(ids)
    df["ID"] = ids
    frames_text = []
    df.fillna(value = 0, inplace = True)
    for index, ID in df["ID"].iteritems():
        try :
            print("opening " + ID)
            print(i, " on ", total)
            data = is_valid(ID)
            d = {} ##
            print(ID + " open")
            for sentence in data :
                for frame in sentence["frames"] :
                    if frame["target"]["name"] in frames_to_keep :
                        df.loc[index,frame["target"]["name"]] += 1
                        print(df.loc[index,frame["target"]["name"]])
                        if frame["target"]["name"] not in d :
                            d[frame["target"]["name"]] = []
                            d[frame["target"]["name"]].append(extract_text(frame))
            frames_text.append(d)
        except :
            print(ID + " could not be opened")
        i += 1
    df.set_index("ID", inplace = True)
    df.to_pickle(os.path.join(data_path,"data.pkl"))

    output_file = open(os.path.join(data_path,"frames_text.pkl"),"wb+")
    pickle.dump(frames_text, output_file)
    output_file.close()

build_matrix()

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
