from json import *
from extraction import *
from df_exploration import *
from build_matrix2 import *
import pandas as pd
from path import *
import pickle

json_path = os.path.join(parent + "/json_abs/")


L = pickle.load(open("out.pkl","rb"))

df = pd.read_pickle("data.pkl")

i = 0

for id in df.index :
    print(get_line(df, id))
    print(L[i])
    i += 1
    
    

