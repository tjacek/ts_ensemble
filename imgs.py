import cv2,numpy as np
import files

class Pipeline(object):
    def __init__(self,transforms):
        self.transforms=transforms

    def __call__(self,frame_i):
        frame_i=self.transforms[0](frame_i)
        for transform_j in self.transforms[1:]:
            frame_i=transform_j(frame_i)
        return frame_i

def transform(in_path,out_path,frame_fun,single_frame=False):
    if(type(frame_fun)==list):
        frame_fun=Pipeline(frame_fun)
    files.make_dir(out_path)
    print(in_path)
    for in_i in files.top_files(in_path):
        out_i=out_path+'/'+in_i.split('/')[-1]
        print(out_i)
        frames=read_frames(in_i)
        if(single_frame):
            new_frames=[frame_fun(frame_j) for frame_j in frames]
        else:
            new_frames=frame_fun(frames)
        save_frames(out_i,new_frames)

def action_img(in_path,out_path,action_fun):
    if(type(action_fun)==list):
        action_fun=Pipeline(action_fun)
    files.make_dir(out_path)
    for in_i in files.top_files(in_path):
        frames=read_frames(in_i)
        action_img_i=action_fun(frames)
        out_i= out_path+'/' + in_i.split('/')[-1]+".png"
        cv2.imwrite(out_i,action_img_i)

def seq_tranform(frame_fun,img_seqs):
    return { name_i:[frame_fun(frame_j) for frame_j in seq_i]
                    for name_i,seq_i in img_seqs.items()}

def read_seqs(in_path):
    seqs={}
    for seq_path_i in files.top_files(in_path):
        frames=read_frames(seq_path_i)
        name_i=seq_path_i.split('/')[-1]
        print(name_i)
        seqs[name_i]=frames
    return seqs    

def save_seqs(seq_dict,out_path):
    files.make_dir(out_path)
    for name_i,seq_i in seq_dict.items():
        seq_path_i=out_path+'/'+name_i
        save_frames(seq_path_i,seq_i)

def read_frames(seq_path_i,as_dict=False):
    if(as_dict):
        return {files.clean_str(path_j):cv2.imread(path_j,cv2.IMREAD_GRAYSCALE)
                    for path_j in files.top_files(seq_path_i)}
    return [ cv2.imread(path_j, cv2.IMREAD_GRAYSCALE)
                for path_j in files.top_files(seq_path_i)]

def save_frames(seq_path_i,seq_i):
    files.make_dir(seq_path_i)
    for j,frame_j in enumerate(seq_i):     
        frame_name_j=seq_path_i+'/'+str(j)+".png"
        cv2.imwrite(frame_name_j,frame_j)

def concat_seq(in_path1,in_path2,out_path):
    seq1,seq2=read_seqs(in_path1),read_seqs(in_path2)
    names=seq1.keys()
    concat_seqs={}
    for name_i in names:
        seq1_i,seq2_i=seq1[name_i],seq2[name_i]
        seq_len=min(len(seq1_i),len(seq2_i))
        seq1_i,seq2_i= seq1_i[:seq_len],seq2_i[:seq_len]
        new_seq_i=np.concatenate( [seq1_i,seq2_i],axis=1)
        concat_seqs[name_i]=new_seq_i
    save_seqs(concat_seqs,out_path)

def concat_frames(in_path1,in_path2,out_path):
    seq1,seq2=read_frames(in_path1,True),read_frames(in_path2,True)
    files.make_dir(out_path)
    for name_i in seq1.keys():
        img0,img1=seq1[name_i],seq2[name_i] 
        new_img_i=np.concatenate([img0,img1],axis=0)
        cv2.imwrite(out_path+'/'+name_i+".png",new_img_i)