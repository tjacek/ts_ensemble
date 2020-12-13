import numpy as np
import os,re

class Name(str):
	def __new__(cls, p_string):
		return str.__new__(cls, p_string)

	def clean(self):
		digits=[ str(int(digit_i)) 
				for digit_i in re.findall(r'\d+',self)]
		return Name("_".join(digits))

	def get_cat(self):
		return int(self.split('_')[0])-1

class Feats(dict):
	def __init__(self, arg=[]):
		super(Feats, self).__init__(arg)

	def __add__(self,feat_i):
		names=common_names(self.keys(),feat_i.keys())
		new_feats=Feats()
		for name_i in names:
			x_i=np.concatenate([self[name_i],feat_i[name_i]],axis=0)
			new_feats[name_i]=x_i
		return new_feats

	def n_cats(self):
		return max(self.get_cats())+1

	def dim(self):
		return list(self.values())[0].shape[0]

	def names(self):
		return sorted(self.keys(),key=natural_keys) 
	
	def split(self,selector=None):
		train,test=split(self,selector)
		return Feats(train),Feats(test)

	def to_dataset(self):
		names=self.names()
		X=np.array([self[name_i] for name_i in names])
		return X,self.get_cats()

	def get_cats(self):
	    return [ name_i.get_cat() for name_i in self.names()]
	
	def transform(self,extractor):
		feat_dict={	name_i: extractor(feat_i)
				for name_i,feat_i in self.items()}
		return Feats(feat_dict)

	def norm(self):
		X,y=self.to_dataset()
		mean=np.mean(X,axis=0)
		std=np.std(X,axis=0)
		std[np.isnan(std)]=1
		std[std==0]=1
		for name_i,feat_i in self.items():
			self[name_i]=(feat_i-mean)/std

	def save(self,out_path):
		lines=[]
		for name_i,feat_i in self.items():
			txt_i=np.array2string(feat_i,separator=",")
			txt_i=txt_i.replace("\n","")
			lines.append("%s#%s" % (txt_i,name_i))
		feat_txt='\n'.join(lines)
		feat_txt=feat_txt.replace('[','').replace(']','')
		feat_txt = feat_txt.replace(' ','')
		file_str = open(out_path,'w')
		file_str.write(feat_txt)
		file_str.close()

def read_feats(in_path):
    if(type(in_path)==list):
        all_feats=[read_feats(path_i) for path_i in in_path]
        return concat_feats(all_feats)
    lines=open(in_path,'r').readlines()
    feat_dict=Feats()
    for line_i in lines:
        raw=line_i.split('#')
        if(len(raw)>1):
            data_i,info_i=raw[0],Name(raw[-1])
            info_i=info_i.clean()
            feat_dict[info_i]=np.fromstring(data_i,sep=',')
    return feat_dict

def concat_feats(all_feats):
	first=all_feats[0]
	for feat_i in all_feats[1:]:
		first+=feat_i
	return first

def common_names(names1,names2):
	return list(set(names1).intersection(set(names2)))

def split(dict,selector=None):
    if(not selector):
        selector=person_selector
    train,test=[],[]
    for name_i in dict.keys():
        if(selector(name_i)):
            train.append((name_i,dict[name_i]))
        else:
            test.append((name_i,dict[name_i]))
    return train,test

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def atoi(text):
    return int(text) if text.isdigit() else text

def top_files(path):
    paths=[ path+'/'+file_i for file_i in os.listdir(path)]
    paths=sorted(paths,key=natural_keys)
    return paths

def person_selector(name_i):
    person_i=int(name_i.split('_')[1])
    return person_i%2==1

if __name__ == "__main__":
	dataset=read_feats('../action/feats')
	print(dataset.dim())
