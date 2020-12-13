import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import feats

class Result(object):
	def __init__(self,y_true,y_pred,names):
#		assert(len(y_true)==len(y_pred))
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

def ensemble(in_path):
	results=[ train_model(path_i,binary=True)
			for path_i in feats.top_files(in_path)]
	votes=np.array([ result_i.as_numpy() 
				for result_i in results])
	votes=np.sum(votes,axis=0)
#	print(votes)
#	votes=np.mean(votes,axis=2)
#	raise Exception(votes.shape)
	return Result(results[0].y_true,votes,results[0].names)

def train_model(dataset,binary=True):
	if(type(dataset)==str):
		dataset=feats.read_feats(dataset)
	dataset.norm()
	train,test=dataset.split()
	model=LogisticRegression(solver='liblinear')
	X_train,y_train=train.to_dataset()
	model.fit(X_train,y_train)
	X_test,y_test=test.to_dataset()
	if(binary):
		y_pred=model.predict(X_test)
	else:
		y_pred=model.predict_proba(X_test)
	return Result(y_test,y_pred,test.names())

def simple_exp(dataset):
	result=train_model(dataset)
	print(result.get_acc())
#	print(classification_report(y_test, y_pred,digits=4))
#	print(accuracy_score(y_test,y_pred))

votes=ensemble('../action/ens/feats')
print(votes.get_acc())