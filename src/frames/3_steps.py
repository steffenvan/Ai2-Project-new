import os
from similarity_utility import most_similar, load_dataframe, load_tfidf_dataframe, most_similar_with_bigrams
from general_functions import open_txt
import time

ref_id = "E83-1018-parscit.130908"

"""    FIRST STEP : From 16000 to 2000 articles using frame counts """
# approx. 3.3 seconds with the current approach

df = load_dataframe()

open_txt(ref_id)
candidates = most_similar(ref_id, df, 2000)
candidates.append(ref_id)
del df

""" SECOND STEP : From 2000 to 50 articles using tfidf and word embeddings """

df = load_tfidf_dataframe().loc[candidates]
print(len(df.index))
candidates = most_similar_with_bigrams(ref_id, df, 50)

print(candidates)
for id in candidates[:4] :
    open_txt(id)
    
""" THIRD STEP : Ranking the 50 articles using the frame content """

pass





