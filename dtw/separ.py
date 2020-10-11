import dtw

def separ(in_path):
    pairs_i=dtw.read(in_path)
    names=pairs_i.names()[0]
    cats=get_cats(names)

def get_cats(names):
    cat_dict=get_cat_dict(names)
    n_cats=get_n_cats(cat_dict)
    cats={i:[] for i in range(n_cats)}
    for name_i,cat_i in cat_dict.items():
        cats[cat_i].append(name_i)
    return cats

def get_cat_dict(names):
    return {name_i:int(name_i.split("_")[0])-1 
	            for name_i in names}

def get_n_cats(cat_dict):
    return max(cat_dict.values())+1	