import numpy as np
import pickle
from dtaidistance import dtw, dtw_ndim
import feats

def make_pairwise_distance(ts_dataset):
    names=list(ts_dataset.keys())
    n_ts=len(names)   
    pairs_dict={ name_i:{name_i:0.0}
                    for name_i in names}
    for i in range(1,n_ts):
        print(i)
        for j in range(0,i):
            name_i,name_j=names[i],names[j]
            distance_ij=dtw_ndim.distance(ts_dataset[name_i],ts_dataset[name_j])
            pairs_dict[name_i][name_j]=distance_ij
            pairs_dict[name_j][name_i]=distance_ij
    return pairs_dict

def save(pairs,out_path):
	with open(out_path, 'wb') as handle:
		pickle.dump(pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read(in_path):
    with open(in_path, 'rb') as handle:
        return pickle.load(handle)

def compute_dtw(in_path,out_path):
	feats.make_dir(out_path)
	paths={ dir_i:"%s/%s" %(out_path,dir_i) for dir_i in ["pairs","feats"]}
	seq_dict=read_seqs(in_path)
	pairs=make_pairwise_distance(seq_dict)
	save(pairs,paths["pairs"])

def read_seqs(in_path):
	return {  path_i.split('/')[-1]:np.loadtxt(path_i,delimiter=',')
	  			for path_i in feats.top_files(in_path)}


def to_feats(in_path,out_path):
	pairs=read(in_path)
	dtw_feats=feats.Feats()
	train,test=feats.split(pairs,names_only=True)
#	raise Exception(train)
	for name_i in pairs.keys():
		dtw_feats[name_i]=np.array([pairs[name_i][name_j] for name_j in train])
#		raise Exception(dtw_feats[name_i].shape)
		print(dtw_feats[name_i].shape)
	dtw_feats.save(out_path)

in_path='../MSR/dtw/skew/seqs'
compute_dtw(in_path,"skew")
to_feats("skew/pairs","skew/dtw")