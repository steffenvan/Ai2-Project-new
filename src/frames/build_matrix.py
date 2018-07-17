from file_content_extraction import *
import spacy
import pickle

curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))

"""
This script allows the creation of two files :
- a data frame, stored as data.pkl, containing for each files the number of occurences of the frames specified in the frames_to_keep list specified below
- a list of dictionnaries, stored as "out.pkl". Each entry of the list is a dictionnary, with a

"""

## TODO: Need to refactor the file to be able to import the files from path file correctly!

frames_to_keep = ['Causation','Increment', 'Means', 'Aggregate','Relational_quantity', 'Evidence','Assessing','Inclusion','Usefulness','Reasoning', 'Cause_to_make_progress','Importance','Desirability', 'Evaluative_comparison', 'Performing_arts', 'Change_position_on_a_scale', 'Trust', 'Position_on_a_scale', 'Predicament', 'Supply', 'Accomplishment']

def get_frame(sentence):
    for frame in sentence["frames"]:
        return frame

def extract_full_frame(json_content):
    for sentence in json_content:
        for frame in sentence["frames"]:
            return frame

def open_json(id):
    path_to_json = json_train_path.joinpath(id)
    with open(path_to_json) as json_file:
        json_data = json.loads(json_file.read())
    return json_data

def is_valid(id):
    json_file = json_train_path.joinpath(id)
    if json_file.exists():
        try:
            return open_json(json_file)
        except:
            print("Invalid json file: ", id)
            return False

def create_frame_content(file_id, data_frame):
    frames_text = []
    valid_file = is_valid(file_id)

    if valid_file:
        d = {} ##
        print(file_id + " open")
        full_frame = extract_full_frame(valid_file)

        for sentence in valid_file:
            temp_frame = get_frame(sentence)

            if temp_frame in frames_to_keep:
                data_frame.loc[index, frame] += 1

                if temp_frame not in d.keys():
                    d[frame] = []
                    test = extract_text(sentence)
                    d.update({frame:test})
        frames_text.append(d)
    else:
        os.remove(json_train_path.joinpath(file_id))

    return frames_text

def list_of_files(path_to_files, extension):
    folder = os.listdir(path_to_files)
    ids = [filename for filename in folder if filename.endswith(extension)]
    return ids

def build_matrix() :
    columns = ["ID"] + frames_to_keep
    df = pd.DataFrame(columns = columns)

    filenames = list_of_files(json_train_path, "json")
    total_file_count = len(filenames)

    df["ID"] = filenames
    print(df["ID"])
    df.fillna(value = 0, inplace = True)

    count = 0
    for index, ID in df["ID"].iteritems():
        print("opening " + ID)
        print(count, " of ", total_file_count)
        frame_content = create_frame_content(ID, df)
        count += 1

    df.set_index("ID", inplace = True)
    df.to_pickle(data_path.joinpath("data.pkl"))

    frames_text_path = data_path.joinpath("frames_text.pkl")
    output_file = open(frames_text_path, "wb+")
    print(output_file)
    pickle.dump(frame_content, output_file)
    output_file.close()
#
# #
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
#
build_matrix()
