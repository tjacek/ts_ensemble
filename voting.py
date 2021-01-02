import numpy as np
import train,feats,files

def ensemble(common_path,binary_path=None,binary=False,agum=True):
	if(type(binary_path)==list):
		datasets=[]
		for binary_i in binary_path:
			datasets+=read_datasets(common_path,binary_i,agum)
	else:
		datasets=read_datasets(common_path,binary_path,agum)
	results=[ train.train_model(data_i,binary)
				for data_i in datasets]
	return voting(results)

def show_acc(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	acc=[ train.train_model(data_i,binary).get_acc()
				for data_i in datasets]
	print(acc)

def read_datasets(common_path,binary_path,agum=False):
	if(common_path):
		common=feats.read_feats(common_path)
	if(binary_path):
		binary=[feats.read_feats(path_i)
					for path_i in files.top_files(binary_path)]
	if(not common_path):
		return binary
	if(not binary_path):
		return [common]
	if(agum):
		return [feats.agum_unify(binary_i,common) for binary_i in binary]
	else:
		return [common+binary_i for binary_i in binary]

def voting(results):
	votes=np.array([ result_i.as_numpy() 
				for result_i in results])
	votes=np.sum(votes,axis=0)
	return train.Result(results[0].y_true,votes,results[0].names)

def select_clf(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	acc=cross_acc(datasets)
	acc= (acc-np.mean(acc))/np.std(acc)
	s_datasets=[data_i 
			for i,data_i in enumerate(datasets)
				if(acc[i]>-0.2)]
	results=[ train.train_model(data_i,binary)
				for data_i in s_datasets]
	print(len(results))
	return voting(results)

def cross_acc(datasets):
    datasets=[data_i.split()[0] for data_i in datasets]
    acc=[]
    for data_i in datasets:
    	new_data_i=feats.Feats()
    	for j,name_j in enumerate(data_i.keys()):
    		new_name="%s_%d" %(name_j.split('_')[0],j)
    		new_data_i[feats.Name(new_name)]=data_i[name_j]
    	result_i=train.train_model(new_data_i,binary=True)
    	acc.append(result_i.get_acc())
    return np.array(acc)

if __name__ == "__main__":
	dataset="3DHOI_agum"
	common_path=['../%s/dtw/max_z/person' % '3DHOI',
             '../%s/dtw/corl/person' % '3DHOI']
#             '../agum/skew/dtw']
	common_path='../3DHOI/agum/lstm/feats'
	votes=ensemble(common_path,'../3DHOI/agum/ens/feats',agum=False)#, '../3DHOI/ens/feats'])
#	votes=select_clf('true/corl.txt',None)#'../3DHOI/ens/feats')
	votes.report()
	print(votes.get_acc())
	print(votes.get_cf())

