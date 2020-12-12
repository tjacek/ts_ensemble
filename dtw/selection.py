import numpy as np
from collections import defaultdict
import dtw

def cat_selection(in_path,out_path):
	selection_template(in_path,out_path,group_by_cat)

def person_selection(in_path,out_path):
	selection_template(in_path,out_path,group_by_person)

def selection_template(in_path,out_path,group_by):
	pairs=dtw.read(in_path)
	names=pairs.names()[1]
	by_person=group_by(names)
	s_names=[get_rep(group_i,pairs) 
				for group_i in by_person.values()]
	print(len(s_names))
	dtw.save_dtw_feats(out_path,pairs,subset=s_names)

def group_by_person(names):
	by_person=defaultdict(lambda:[])
	for name_i in names:
		id_i="_".join(name_i.split('_')[:2])
		by_person[id_i].append(name_i)
	return by_person

def group_by_cat(names):
	by_person=defaultdict(lambda:[])
	for name_i in names:
		id_i=name_i.split('_')[0]
		by_person[id_i].append(name_i)
	return by_person


def get_rep(group,pairs):
	dist=[[ pairs(name_i,name_j)
			for name_i in group]
				for name_j in group]
	dist=np.array(dist)
	dist=np.sum(dist,axis=0)
	return  group[np.argmax(dist)]