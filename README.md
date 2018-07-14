# Ai2-Project-New
New repository
=================
We have created a new repository due to LFS (large file) problems.

## Setup

note : please use the following set up to make sure everything works fine :
- Ensure that the data folder is one folder above the relative path as the repository.

## Data

This folder should contain the files and folders found from [this link](https://drive.google.com/drive/folders/1c2yp5YUgS-OlUb5p7xIDLPlMMOVLsbIX).
There should be a folder in the parent directory of the project repository named `data`.
#### Retraining Semafor
There is a large zip file containing the training data called `train.zip`. This contains the ACL data that we have used to train semafor. Any files in `xml` be used.  Place the unzipped folder in the subdirectory of `data`.

#### Test data
Test the program on untrained data by creating a folder `test` that contains the `xml` files to be tested. Run semafor on these to create the `.json` files. Next run `test.py` (We need to rename this) to find the most similar articles.    

The gensim model should reside in the root directory of `data`
- `mymodel.gsm`

# Project tree
```
|-- project/
|   -- data/
|       -- corpus_words.txt
|       -- dico.txt
|       -- mymodel.gsm
|       -- test/
|       -- train/
|   -- Ai2-Project-new/
|       -- src/
|           |-- Dockerfile
|           |-- __init__.py
|           |-- visualization/
|           |-- frames/
|           |-- preprocessing/
|           |-- path.py
```

## Similarity values:
An abstract and its similarity values (dice, jaccard, cosine, WMD) can be shown by running `frame_similarity.py`.

# Note on src files :

`build_matrix.py` :
build_matrix() function for creating the dataframe counting the frames appearing in each documents. Only the relevant frames (see the list at the top of the script) are taken into an account.

`explore.py`:
This file contains useful data exploration functions.

`extraction.py`:
Contains functions to extract specific sections from text or json files.

`frame_similarity.py`:
Will output the similarity measures comparing a reference document with other documents.

`pipeline.py`:
Run this script every time you added new `xml` files to the `data/xml/` folder. It will run the preprocessing steps and create the associated `txt` and `json` files.

`preprocess_functions.py`:
Contains all the functions called by pipeline.py and article.py.
