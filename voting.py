import numpy as np
import train,feats

def ensemble(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	results=[ train.train_model(data_i,binary)
				for data_i in datasets]
	return voting(results)

def show_acc(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	acc=[ train.train_model(data_i,binary).get_acc()
				for data_i in datasets]
	print(acc)

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

def select_clf(common_path,binary_path=None,binary=False):
	datasets=read_datasets(common_path,binary_path)
	acc=cross_acc(datasets)
	acc= (acc-np.mean(acc))/np.std(acc)
	s_datasets=[data_i 
			for i,data_i in enumerate(datasets)
				if(acc[i]>-1)]
	results=[ train.train_model(data_i,binary)
				for data_i in s_datasets]
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


common_path=['../agum/corl/person',
             '../agum/max_z/person']
#             '../agum/skew/person']
votes=select_clf(common_path,'../action/ens/feats')
votes.report()
print(votes.get_acc())
