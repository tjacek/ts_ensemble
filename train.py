import numpy as np,os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import feats,files

class Result(object):
	def __init__(self,y_true,y_pred,names):
		self.y_true=y_true
		self.y_pred=y_pred
		self.names=names

	def as_numpy(self):
		if(self.y_pred.ndim==2):
			return self.y_pred
		else:			
			print(len(self.y_pred))
			n_cats=np.amax(self.y_true)+1
			votes=np.zeros((len(self.y_true),n_cats))
			for  i,vote_i in enumerate(self.y_pred):
				votes[i,vote_i]=1
			return votes
	
	def as_labels(self):
		if(self.y_pred.ndim==2):
			pred=np.argmax(self.y_pred,axis=1)
		else:
			pred=self.y_pred
		return self.y_true,pred

	def get_acc(self):
		y_true,y_pred=self.as_labels()
		return accuracy_score(y_true,y_pred)

	def report(self):
		y_true,y_pred=self.as_labels()
		print(classification_report(y_true, y_pred,digits=4))

	def get_cf(self,out_path=None):
		y_true,y_pred=self.as_labels()
		cf_matrix=confusion_matrix(y_true,y_pred)
		if(out_path):
			np.savetxt(out_path,cf_matrix,delimiter=",",fmt='%.2e')
		return cf_matrix

def train_model(dataset,binary=True):
	if(type(dataset)==str or type(dataset)==list):
		dataset=feats.read_feats(dataset)
	dataset.norm()
	train,test=dataset.split()
	model=LogisticRegression(solver='liblinear')
	X_train,y_train=train.to_dataset()
	print(X_train.shape)
	X_train=np.nan_to_num(X_train)
	model.fit(X_train,y_train)
	X_test,y_test=test.to_dataset()
	X_test=np.nan_to_num(X_test)

	if(binary):
		y_pred=model.predict(X_test)
	else:
		y_pred=model.predict_proba(X_test)
	return Result(y_test,y_pred,test.names())

def simple_exp(dataset):
	if(os.path.isdir(dataset)):
		dataset=files.top_files(dataset)
	result=train_model(dataset)
	result.report()
	print(result.get_acc())

if __name__ == "__main__":
	simple_exp("../MSR/ens/feats")