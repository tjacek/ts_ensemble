import matplotlib.pyplot as plt
import files
from dtw.separ import get_cats

def show_ts(in_path,out_path):
	seqs=files.get_seqs(in_path)
	dim=list(seqs.values())[0].shape[-1]
	cats=get_cats(seqs.keys())[0]
	files.make_dir(out_path)
	for cat_i,names_i in cats.items():
		in_i="%s/%d" % (out_path,cat_i)
		files.make_dir(in_i) 		
		for name_ij in names_i:
			out_ij="%s/%s" % (in_i,name_ij)
			seq_ij=seqs[name_ij]
			for ts_k in seq_ij.T:
				plt.plot(ts_k)
			plt.savefig(out_ij)
			plt.clf()
			plt.close()