# Ai2-Project-New
New repository
=================
We have created a new repository due to LFS (large file) problems.

## Setup

note : please use the following set up to make sure everything works fine :
- Ensure that the data folder is one folder above the relative path as the repository.

## Data

This folder should contain the files and folders. They can be retrieved [here](https://drive.google.com/drive/folders/1c2yp5YUgS-OlUb5p7xIDLPlMMOVLsbIX): 

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
