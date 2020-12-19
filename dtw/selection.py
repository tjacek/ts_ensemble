import numpy as np
import pickle
from collections import defaultdict
import feats

class DTWPairs(object):
	def __init__(self, pairs):
		self.pairs = pairs

	def group_by(self,get_id):
		train,test=self.split()
		groups=defaultdict(lambda:[])
		for name_i in train:
			id_i=get_id(name_i)
			groups[id_i].append(name_i)
		return groups

	def distance_matrix(self,group):
		dist=[[ self.pairs[name_i][name_j]
				for name_i in group]
					for name_j in group]
		return np.array(dist)

	def get_vector(self,name_i,train):
		return np.array([ self.pairs[name_i][name_j] 
							for name_j in train])

	def split(self):
		train,test=[],[]
		for name_i in self.pairs.keys():
			if(feats.person_selector(name_i)):
				train.append(name_i)
			else:
				test.append(name_i)
		return train,test

	def selection(self,get_id,selector):
		groups=self.group_by(get_id)
		selected=[]
		for group_i in groups.values():
			selected.append(selector(group_i,self))
		return  selected

	def with_test(self,names):
		train,test=self.split()
		return test+names

def read(in_path):
    with open(in_path, 'rb') as handle:
        return DTWPairs(pickle.load(handle))

def cat_id(name_i):
	return name_i.split('_')[0]

def person_id(name_i):
	return "_".join(name_i.split('_')[:2])

def center_selector(group_i,dtw_pairs):
	dist_i=dtw_pairs.distance_matrix(group_i)
	dist_i=np.sum(dist_i,axis=0)
	return group_i[np.argmin(dist_i)]

#def true_one_shot(in_path,out_path):
#	dtw_pairs=read(in_path)
#	names=dtw_pairs.selection(cat_id,center_selector)
#	full=dtw_pairs.with_test(names) #test+names
#	s_feats=feats.Feats()
#	for name_i in full:
#		s_feats[name_i]=dtw_pairs.get_vector(name_i,names)
#	s_feats.save(out_path)

def one_shot(main,add,out_path):
	dtw_pairs=read(main)
	names=dtw_pairs.selection(person_id,center_selector)
	full=dtw_pairs.with_test(names)
	pairs=[dtw_pairs,read(add)] if(add) else [dtw_pairs]
	s_feats=feats.Feats()
	for name_i in full:
		vectors=[ pairs_i.get_vector(name_i,names) 
					for pairs_i in pairs]
#		v0=dtw_pairs.get_vector(name_i,names)
#		v1=add_pairs.get_vector(name_i,names)
		s_feats[name_i]=np.concatenate(vectors,axis=0)
	s_feats.save(out_path)

#true_one_shot("../MHAD/dtw/max_z/pairs","MHAD/simple/max_z")
main="../agum/corl/pairs"
add="../agum/max_z/pairs"
out="few_shot/MHAD/simple"
one_shot(add,None,"few_shot/MSR/simple/max_z")
