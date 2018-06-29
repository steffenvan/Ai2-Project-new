import pandas as pd
from extraction import *

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
# 
# df = pd.read_pickle("data.pkl")    

def get_line(df, id) :
    return df.loc[id,:]
    
# print(get_line(df, "W05-0503-parscit.130908.json"))
# article = list(df.index.values)[0]
# 
# for file in list(df.index.values) :
# 
# 
# 
# for file in list(df.index.values) :
#     print(list(get_line(df, file)))
    
    
def distance(L,M) :
    assert len(L)==len(M)
    return sum([L[i]>0 and M[i]>0 for i in range(len(L))])
    
print(distance([1,0,3],[1,1,0]))
    
    