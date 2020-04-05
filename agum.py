import numpy as np
import files,spline

class WrapSeq(object):
    def __init__(self,seq_len=128,sub_seq=32):
        self.seq_len=seq_len
        self.sub_seq_len=sub_seq
        self.main_spline=spline.SplineUpsampling(self.seq_len)
        self.warps=[start_warp,end_warp]
        self.splines=[spline.SplineUpsampling(2*sub_seq),spline.SplineUpsampling(sub_seq/2)]

    def __call__(self,seq_i):
        agum=[]
        for warp_j in self.warps:
            for spline_k in self.splines:
                agum.append( warp_j(seq_i,self.sub_seq_len,spline_k)) 
        agum=[self.main_spline(new_seq_i)
                for new_seq_i in agum]
        return agum

def start_warp(seq_i,start,spline):
    sub_i,rest_i=seq_i[:start],seq_i[start:]
    sub_i=spline(sub_i)
    return np.concatenate([sub_i,rest_i])

def end_warp(seq_i,end,spline):
    cut_point= seq_i.shape[0]-end
    sub_i,rest_i=seq_i[:cut_point],seq_i[cut_point:]
    rest_i=spline(rest_i)
    return np.concatenate([sub_i,rest_i])

def scale_agum(data_i):
    return [scale_j*data_i for scale_j in [0.5,2.0]]

def apply_agum(in_path,out_path):
    seq_dict=files.get_seqs(in_path)
    train,test=files.split(seq_dict)
    agum_train=[]
    agum=WrapSeq()
    for name_i,seq_i in seq_dict.items():
        new_seqs= agum(seq_i) #scale_agum(seq_i)
        for j,seq_j in enumerate(new_seqs):
            name_j="%s_%d" % (name_i,j)
            agum_train.append((name_j,seq_j))
    agum_train=dict(agum_train)
    files.save_seqs(agum_train,out_path)
    print(len(agum_train))  

apply_agum('test','agum')