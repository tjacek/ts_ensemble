import numpy as np
import scipy.stats
import files

def compute_stats(in_path,out_path):
    seqs=files.get_seqs(in_path)
    feats={ name_i:EBTF(seq_i) 
            for name_i,seq_i in seqs.items()}
    feats=files.FeatDict(feats)
    X,y=feats.as_dataset()
    names=list(feats.keys())
    files.save_feats(X,names,out_path)

def EBTF(feat_i):
    if( len(feat_i.shape)>1):
        ts=[EBTF(feat_ij) for feat_ij in feat_i.T]	
        return np.concatenate(ts)
    if(np.all(feat_i==0)):
        return [0.0,0.0,0.0,0.0]
    return [np.mean(feat_i),np.std(feat_i),
    	                scipy.stats.skew(feat_i),time_corl(feat_i)]

def time_corl(feat_i):
    n_size=feat_i.shape[0]
    x_i=np.arange(float(n_size),step=1.0)
    return scipy.stats.pearsonr(x_i,feat_i)[0]

compute_stats("full/seqs","full/feats.txt")