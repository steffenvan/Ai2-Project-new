# Ai2-Project-New
New repository
=================
We have created a new repository due to LFS problems.

## Setup

note : please use the following set up to make sure everything works fine :

- Have your Ai2-Project saved in the same directory as the root directory of Semafor
- In the `src` folder, create `path.py`, which should contain the absolute path to the `Ai2-Project` folder.

## Semafor
- So far we have managed to run the frame-semantic parser on the output of the dummy
file `src/article.xml`.
### Only new lines
It is important to note that the parser only works for files with no em
### Results
To see the result from parsing the `article.xml` file using Semafor, there is
file located in `results/temp-res.txt` to show the immediate frame-semantic information.

# src files :

`pipeline.py`:
Run this script every time you added new `xml` files to the `data/xml/` folder. It will run the preprocessing steps and create the associated `txt` and `json` files.

`explore.py`:
This file contains useful data exploration functions.

`extraction.py`:
Contains functions to extract specific sections from text or json files.

`preprocess_functions.py`:
Contains all the functions called by pipeline.py and article.py.
