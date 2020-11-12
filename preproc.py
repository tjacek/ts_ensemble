import cv2
import numpy as np
import imgs,files

def scaled_frames(in_path,out_path):
    imgs.transform(in_path,out_path,scale,single_frame=True)

def smooth_frames(in_path,out_path):
    fun=[gauss_helper,scale]
    imgs.transform(in_path,out_path,gauss_helper,single_frame=True)

def scale(img_i):
    dim=(64,64)
    inter=cv2.INTER_CUBIC
    return cv2.resize(img_i,dim,inter)

def gauss_helper(img_i):
    return cv2.GaussianBlur(img_i, (9, 9), 0)

def norm_seqs(in_path,out_path):
	seqs=files.get_seqs(in_path)
	seq_max=np.array([np.amax(seq_i,axis=0)
				for name_i,seq_i in seqs.items()])
	seq_max= np.amax(seq_max,axis=0)
	new_seqs={}
	for name_i,seq_i in seqs.items():
		seq_i=[  ts_j/seq_max[j] 
				for j,ts_j in enumerate(seq_i.T)]
		seq_i=np.array(seq_i)
		new_seqs[name_i]=seq_i
	files.make_dir(out_path)
	for name_i,seq_i in new_seqs.items():
		out_i="%s/%s" % (out_path,name_i)
		np.save(out_i,seq_i)

if __name__=="__main__":
    norm_seqs("skeleton/seqs","new_seqs")