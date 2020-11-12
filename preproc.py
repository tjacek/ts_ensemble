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
	fun= sigma_norm(seqs)
	new_seqs={}
	for name_i,seq_i in seqs.items():
		seq_i=[ fun(ts_j,j) 
				for j,ts_j in enumerate(seq_i.T)]
		seq_i=np.array(seq_i).T
		new_seqs[name_i]=seq_i
	files.make_dir(out_path)
	for name_i,seq_i in new_seqs.items():
		out_i="%s/%s" % (out_path,name_i)
		np.save(out_i,seq_i)

def norm_max(seqs):
	seq_max=np.array([np.amax(seq_i,axis=0)
				for name_i,seq_i in seqs.items()])
	seq_max=np.amax(seq_max,axis=0)
	return lambda ts_j,j: ts_j/seq_max[j]

def sigma_norm(seqs):
	all_seqs=np.concatenate(list(seqs.values()),axis=0)
	mean_seq=np.mean(all_seqs,axis=0)
	std_seq=np.std(all_seqs,axis=0)
	return lambda ts_j,j: (ts_j-mean_seq[j])/std_seq[j]

if __name__=="__main__":
    norm_seqs("skeleton/seqs","skeleton2/seqs")