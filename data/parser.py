from ast import literal_eval
from pathlib import Path
import pathlib

import numpy as np


def dumb_parse_c_array(header: Path):
    'Receives a c header file and parses the first c 2d array it finds into a numpy array'
    with open(header, 'r+') as headerfile:
        headertext = headerfile.read(-1)
        headertext = headertext.split('=')[1].replace("{", "[").replace("}", "]").replace(";", "")
        newtext = ""
        for line in headertext.splitlines():
            splitline = line.strip().split("   ")
            newtext += splitline[0].strip() + "\n"
        list = literal_eval(newtext)
        newlist = []
        for char in list:
            newchar = []
            for row in char:
                for bitnum in range(4, -1, -1):
                    bit = (row >> bitnum) & 1
                    newchar.append(bit)
            newlist.append(newchar)
        return np.asarray(newlist)

if __name__ == '__main__':
    print(dumb_parse_c_array(pathlib.Path(__file__).parent.resolve() / 'font.h'))