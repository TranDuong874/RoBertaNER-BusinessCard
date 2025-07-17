import numpy as np

def label_to_id(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[label] = i
    return dic
    
def id_to_label(label_list):
    dic = {}
    for i, label in enumerate(label_list):
        dic[i] = label
    return dic
