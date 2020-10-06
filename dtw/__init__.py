import numpy as np
import pickle
import files
#import unify,filtr,feats

class DTWPairs(object):
    def __init__(self,pairs):
        self.pairs=pairs
    
    def names(self):
        all_names=list(self.pairs.keys())
        train,test=files.split(self.pairs)
        return all_names,train,test

    def to_features(self):
        all_names,train,test=self.names()
        def dtw_helper(name_i):
            return np.array([ self.pairs[name_i][name_j] 
                                for name_j in train])
        return {name_i:dtw_helper(name_i) for name_i in all_names} 

#        X=[ dtw_for name_i in all_names]
#        files.save_feats(X,all_names,out_path)   
#        return feats.from_dict(feat_dict) 

    def distances(self,test,train):
        dist=[[self.pairs[name_i][name_j] 
                for name_i in test]
                    for name_j in train]
        return np.array(dist)

    def save(self,out_path):
        with open(out_path, 'wb') as handle:
            pickle.dump(self.pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

def make_dtw_feats(dir_path):
    seq_path=dir_path+ "/seqs"
    pair_path=dir_path+ "/pairs"
    dtw_path=dir_path+ "/dtw"
    ts_dataset=files.get_seqs(seq_path)

    make_pairwise_distance(ts_dataset).save(pair_path)
    dtw_pairs=read(pair_path)
    dtw_feats=dtw_pairs.to_features()
    names=list(dtw_feats.keys())
    X=[ dtw_feats[name_i] 
        for name_i in names]
    files.save_feats(X,names,dtw_path)
#    dtw_feats.save(dtw_path)

def read(in_path):
    with open(in_path, 'rb') as handle:
        return DTWPairs(pickle.load(handle))

def make_pairwise_distance(ts_dataset):
    names=list(ts_dataset.keys())
    n_ts=len(names)   
    pairs_dict={ name_i:{name_i:0.0}
                    for name_i in names}
    for i in range(1,n_ts):
        print(i)
        for j in range(0,i):
            name_i,name_j=names[i],names[j]
            distance_ij=dtw_metric(ts_dataset[name_i],ts_dataset[name_j])
            pairs_dict[name_i][name_j]=distance_ij
            pairs_dict[name_j][name_i]=distance_ij
    return DTWPairs(pairs_dict)

def dtw_metric(s,t):
    dtw,n,m=prepare_matrix(s,t)
    for i in range(1,n+1):
        for j in range(1,m+1):
            t_i,t_j=s[i-1],t[j-1]
            diff=t_i-t_j
            cost= np.dot(diff,diff)          
            dtw[i][j]=cost+min([dtw[i-1][j],dtw[i][j-1],dtw[i-1][j-1]])
    return np.sqrt(dtw[n][m])

def prepare_matrix(s,t):
    n=len(s)
    m=len(t)
    cost_matrix=np.zeros((n+1,m+1),dtype=float)
    for i in range(1,n+1):
        cost_matrix[i][0]=np.inf
    for i in range(1,m+1):
        cost_matrix[0][i]=np.inf
    return cost_matrix,n,m