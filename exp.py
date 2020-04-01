import numpy as np
import deep,files

def simple_exp(in_path):
    seq_dict=files.get_seqs(in_path)
    prepare_data(seq_dict)
#    deep.make_conv(params)

def prepare_data(seq_dict):
    names=seq_dict.keys()
    data=list(seq_dict.values())
    data=np.array(data)
    params={'ts_len':data.shape[1], 'n_feats': data.shape[2]}
    return names,data,params

simple_exp("test")