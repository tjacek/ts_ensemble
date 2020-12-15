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
		for name_i in test:
			id_i=get_id(name_i)
			groups[id_i].append(name_i)
		return groups

	def distance_matrix(self,group):
		dist=[[ self.pairs[name_i][name_j]
				for name_i in group]
					for name_j in group]
		return np.array(dist)

	def split(self):
		train,test=[],[]
		for name_i in self.pairs.keys():
			if(feats.person_selector(name_i)):
				train.append(name_i)
			else:
				test.append(name_i)
		return train,test

def cat_id(name_i):
	return name_i.split('_')[0]

def read(in_path):
    with open(in_path, 'rb') as handle:
        return DTWPairs(pickle.load(handle))

def selection(in_path):
	dtw_pairs=read(in_path)
	groups=dtw_pairs.group_by(cat_id)
	selected=[]
	for group_i in groups.values():
		dist_i=dtw_pairs.distance_matrix(group_i)
		dist_i=np.sum(dist_i,axis=0)
		selected.append(group_i[ np.argmin(dist_i)])
	print(selected)

selection('../agum/corl/pairs')
#print(pairs.group_by(cat_id))