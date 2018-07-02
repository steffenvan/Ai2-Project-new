# Ai2-Project-New
New repository
=================
We have created a new repository due to LFS (large file) problems.

## Setup

note : please use the following set up to make sure everything works fine :

- Have your Ai2-Project saved in the same directory as the root directory of Semafor
- In the `src` folder, modify `path.py` to contain the absolute path to the `Ai2-Project-new` folder.

# Similarity

We can compute a simple similarity of the text of the common frames of an abstract
and the other abstracts.

## Similarity values:
See the temporary results by running `frame_similarity.py`. 

# Note on src files :

`pipeline.py`:
Run this script every time you added new `xml` files to the `data/xml/` folder. It will run the preprocessing steps and create the associated `txt` and `json` files.

`explore.py`:
This file contains useful data exploration functions.

`extraction.py`:
Contains functions to extract specific sections from text or json files.

`preprocess_functions.py`:
Contains all the functions called by pipeline.py and article.py.

`build_matrix.py` :
build_matrix() function for creating the dataframe counting the frames appearing in each documents. Only the relevant frames (see the list at the top of the script) are taken into an account.
