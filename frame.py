import numpy as np
from scipy.stats import skew,pearsonr
import imgs,files

class Extract(object):
    def __init__(self,funcs=None):
        if(not funcs):
            funcs=[max_z,skew_feat,corl,std_feat]
        self.funcs=funcs

    def __call__(self,frames):
        pclouds=prepare_pclouds(frames)
#        pclouds_fun=[max_z,skew_feat,corl,std_feat]
        feats=make_feat_seq(pclouds,self.funcs)
        frame_fun=[]#area]
        frame_feats=make_feat_seq(frames,frame_fun)
        full=np.concatenate(feats+frame_feats,axis=1)
        return full

def compute(in_path,out_path,upsample=False,feats=None):
    seq_dict=imgs.read_seqs(in_path)
    files.make_dir(out_path)
    extract=Extract([max_z])
    for name_i,seq_i in seq_dict.items():
        feat_seq_i=extract(seq_i)
        name_i=name_i.split('.')[0]+'.txt'
        out_i=out_path+'/'+name_i
        np.savetxt(out_i,feat_seq_i,delimiter=',')

def extract(frames):
    pclouds=prepare_pclouds(frames)
    pclouds_fun=[max_z,skew_feat,corl,std_feat]
    feats=make_feat_seq(pclouds,pclouds_fun)
    frame_fun=[area]
    frame_feats=make_feat_seq(frames,frame_fun)
    full=np.concatenate(feats+frame_feats,axis=1)
    return full

def make_feat_seq(frames,funcs):
    return [np.array([fun_j(frame_i)
                for frame_i in frames])
                    for fun_j in funcs]

def prepare_pclouds(frames):
    frames=z_normalize(frames)
    pclouds=[ nonzero_points(frame_i) for frame_i in frames]
#    pclouds=out_normalize(pclouds)
    return pclouds

def z_normalize(frames):
    frames=np.array(frames)
    frames[frames!=0]-= (np.amin(frames[frames!=0])-1)
    frames=frames/np.amax(frames)
    return frames 

def out_normalize(pclouds):
    pclouds=center_normalize(pclouds)
    return outliner(pclouds)

def center_normalize(pclouds):
    center=center_of_mass(pclouds)
    pclouds=[(pcloud_i.T-center).T for pcloud_i in pclouds ]
#    pclouds=[(pcloud_i.T/ np.amax(pcloud_i,axis=1)).T 
#                 for pcloud_i in pclouds ]
    return pclouds

def center_of_mass(pclouds):
    arr_i=np.concatenate(pclouds,axis=1)
    return np.mean(arr_i,axis=1)

def get_max(pclouds):
    return np.amax([ np.amax(pcloud_i,axis=1) 
                      for pcloud_i in pclouds],axis=0)

def outliner(pclouds):
    out=[ pcloud_i *pcloud_i*np.sign(pcloud_i) for pcloud_i in pclouds ]
    pc_max=get_max(out)
    pclouds=[ (pcloud_i.T/pc_max).T for pcloud_i in pclouds]
    return pclouds

def area(frame_i):
    return [np.count_nonzero(frame_i)/np.prod(frame_i.shape)]

def max_z(points):  
    max_index=np.argmax(points[2])
    extr=points[:,max_index]
#    raise Exception(extr)
    return [extr[0],extr[1]]

def std_feat(points):
    return list(np.std(points,axis=1))

def skew_feat(points):
    skew_i=list(skew(points,axis=1))
    return skew_i

def corl(points):
    x,y,z=points[0],points[1],points[2]
    return [pearsonr(x,y)[0],pearsonr(z,y)[0],pearsonr(x,z)[0]]

def nonzero_points(frame_i):
    xy_nonzero=np.nonzero(frame_i)
    z_nozero=frame_i[xy_nonzero]
    xy_nonzero,z_nozero=np.array(xy_nonzero),z_nozero
    x= xy_nonzero[0] / frame_i.shape[0]
    y= xy_nonzero[1] / frame_i.shape[1]
    return np.array([x,y,z_nozero])

#if __name__=="main":
print("OK")
compute("box","max_z2/seqs")