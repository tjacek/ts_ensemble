import numpy as np
import deep,files

def simple_exp(in_path):
    seq_dict=files.get_seqs(in_path)
    train,test=files.split(seq_dict)
    X,y,names,params=prepare_data(train)
#    print(params)
    model=deep.make_conv(params)
    model.fit(X,y)
#    print(dir(models))

def prepare_data(seq_dict):
    names=seq_dict.keys()
    data=list(seq_dict.values())
    X=np.array(data)
    y=[ int(name_i.split("_")[0])-1 for name_i in names] 
    params={'ts_len':X.shape[1], 'n_feats': X.shape[2],
                'n_cats':max(y)+1}
    return X,y,names,params


simple_exp("test")