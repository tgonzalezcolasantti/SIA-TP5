from ast import literal_eval
from pathlib import Path
import pathlib
from typing import List, Tuple

import numpy as np


def dumb_parse_c_array(header: Path)-> Tuple[np.ndarray, List[str]]:
    'Receives a c header file and parses the first c 2d array it finds into a numpy array'
    with open(header, 'r+') as headerfile:
        headertext = headerfile.read(-1)
        headertext = headertext.split('=')[1].replace("{", "[").replace("}", "]").replace(";", "")
        newtext = ""
        labels = []
        for line in headertext.splitlines():
            splitline = line.strip().split("   ")
            newtext += splitline[0].strip() + "\n"
            if len(splitline) > 1:
                labels.append(splitline[1].split(", ")[1])


        font_list = literal_eval(newtext)
        newlist = []
        for char in font_list:
            newchar = []
            for row in char:
                for bitnum in range(4, -1, -1):
                    bit = (row >> bitnum) & 1
                    newchar.append(bit)
            newlist.append(newchar)
        return np.asarray(newlist), labels

if __name__ == '__main__':
    print(dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / 'font.h'))