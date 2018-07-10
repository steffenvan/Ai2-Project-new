import os

# root = os.path.abspath(os.getcwd())

root = os.path.abspath(os.path.join(os.path.abspath(os.getcwd()), os.pardir)) # Ai2-Project-new

parent = os.path.abspath(os.path.join(os.path.abspath(root), os.pardir)) # parent dir containing Ai2, semafor, json_abs

data_path = os.path.join(parent, "data/")

abs_path = os.path.join(data_path, "json_abs/")