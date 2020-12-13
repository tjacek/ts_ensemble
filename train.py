import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import feats

class Result(object):
	def __init__(self,y_true,y_pred):
		self.y_true=y_true
		self.y_pred=y_pred

	def as_labels(self):
		if(self.y_pred.ndim==2):
			pred=np.argmax(self.y_pred,axis=1)
		else:
			pred=self.y_pred
		return self.y_true,pred

	def get_acc(self):
		y_true,y_pred=self.as_labels()
		return accuracy_score(y_true,y_pred)

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
	return Result(y_test,y_pred)

def simple_exp(dataset):
	result=train_model(dataset)
	print(result.get_acc())
#	print(classification_report(y_test, y_pred,digits=4))
#	print(accuracy_score(y_test,y_pred))

simple_exp('../action/feats')