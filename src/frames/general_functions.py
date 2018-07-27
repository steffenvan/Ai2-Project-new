from pathlib import Path
import sys
import os
curr_dir = Path.cwd()
curr_file = curr_dir.joinpath(sys.argv[0])
sys.path.append(str(Path(curr_file).parents[1]))
from path import *

def open_txt(id, train = True) :
    os.system("open "+ str(get_path(id, False, train)))
    