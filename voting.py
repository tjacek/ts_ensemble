import numpy as np
import train,feats

def ensemble(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	results=[ train.train_model(data_i,binary)
				for data_i in datasets]
	return voting(results)

def read_datasets(common_path,binary_path):
	if(common_path):
		common=feats.read_feats(common_path)
	if(binary_path):
		binary=[feats.read_feats(path_i)
					for path_i in feats.top_files(binary_path)]
	if(not common_path):
		return binary
	if(not binary_path):
		return [common]
	return [common+binary_i for binary_i in binary]

def voting(results):
	votes=np.array([ result_i.as_numpy() 
				for result_i in results])
	votes=np.sum(votes,axis=0)
	return train.Result(results[0].y_true,votes,results[0].names)

common_path=['../agum/corl/person',
                '../agum/max_z/person']
votes=ensemble(common_path,'../action/ens/feats')
votes.report()
print(votes.get_acc())
