import numpy as np
import dtw

def separ(in_path):
    pairs_i=dtw.read(in_path)
    names=pairs_i.names()[0]
    cats,cat_dict=get_cats(names)
    cats_clos=get_cats_clos(cats)
    print(cats_clos.keys())
    for name_i in names:
        j=cat_dict[name_i]
        cat_j= cats[j]
        same= pairs_i.dtw_vector(name_i,cat_j)
        cat_clos_j= cats_clos[j]
        other=pairs_i.dtw_vector(name_i,cat_clos_j)
        print("***********")
        min_i=np.amin(other)
        max_i=np.amax(same)
        if(max_i<min_i):
            print(name_i)

def get_cats_clos(cats):
    n_cats=len(cats.keys())
    cats_clos={i:[] for i in range(n_cats)}
    for i in range(n_cats):
        for j in range(n_cats):
            if(i!=j):
                cats_clos[i]+=cats[j]	
    return cats_clos

def get_cats(names):
    cat_dict=get_cat_dict(names)
    n_cats=get_n_cats(cat_dict)
    cats={i:[] for i in range(n_cats)}
    for name_i,cat_i in cat_dict.items():
        cats[cat_i].append(name_i)
    return cats,cat_dict

def get_cat_dict(names):
    return {name_i:int(name_i.split("_")[0])-1 
	            for name_i in names}

def get_n_cats(cat_dict):
    return max(cat_dict.values())+1	