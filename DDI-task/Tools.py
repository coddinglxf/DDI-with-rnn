__author__ = 'Administrator'

from Document import Document

import time
import os


# load data from a path
def load_data(path=""):
    time.sleep(0.5)
    print("start to load data from path----->", path)
    time.sleep(0.5)
    file_list = os.listdir(path)
    sentences = list()

    for i in range(len(file_list)):
        filename = file_list[i]
        current_path = os.path.join(path, filename)
        document = Document(filename=current_path)
        for sentence in document.sentence_list:
            sentences.append(sentence)

    return sentences


def isNumber(inputString):
    pass

# load_data(path="data//drug_and_medline_tidy//")
