import pandas as pd
from extraction import *
from path import *

json_path = os.path.join(parent + "/json_abs/")

def identical_words(L,M) :   # given two lists of words, computes how many words appear in both lists
    count = 0
    
    if len(M) > len(L) :
        for elt in L :
            if elt in M :
                count += 1
    else :
        for elt in M :
            if elt in L :
                count += 1
                
    return count
   

def get_line(df, id) :             # returns a list, each number representing to the number of occurences of the correspounding frame in the build_matrix2/to_keep list
    return (df.loc[id,:]).tolist()
    
    
def similarity(L,M) :     # computes the number elements in L and M which are non-zero at the same time
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])
    

# text = open("out.txt").read().split()
# 
# print(text)

# df = pd.read_pickle("data.pkl")
# 
# reference = get_line(df, df.index[0])
# out = json.load(open(json_path + str(df.index[0])))
# for sentence in out :
#     for frame in sentence["frames"] :
#         print(extract_text(frame))
# 
# print(df.index[0])
# 
# article1 = 
# 
# for id in df.index[1:] :
#     sim = similarity(reference, get_line(df,id))
#     if sim > 4 :
#         print(id)
#         out = json.load(open(json_path + str(id)))
#         for sentence in out :
#             for frame in sentence["frames"] :
#                 if frame["target"]["name"] in df.columns :
#                     print(extract_text(frame))
    
    