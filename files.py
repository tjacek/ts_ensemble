import numpy as np
import os,re

class Name(str):
    def __new__(cls, p_string):
        return str.__new__(cls, p_string)

    def clean(self):
        digits=[ str(int(digit_i)) 
                for digit_i in re.findall(r'\d+',self)]
        return Name("_".join(digits))

    def get_id(self):
        return "_".join(self.split('_')[:3])

    def get_cat(self):
        return int(self.split('_')[0])-1

def split(dict,selector=None,names_only=False):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in dict.keys():
        example_i= name_i if(names_only) else (name_i,dict[name_i])
        if(selector(name_i)):
            train.append(example_i)
        else:
            test.append(example_i)
    return train,test

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def make_dir(path):
    if(not os.path.isdir(path)):
        os.mkdir(path)

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1

def get_seqs(in_path):
    return { path_i.split('/')[-1]:np.load(path_i) 
                for path_i in top_files(in_path)}