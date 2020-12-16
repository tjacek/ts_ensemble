import numpy as np
import matplotlib.pyplot as plt
import voting,train

def acc_curve(common_path,binary_path=None,binary=False):
	datasets=voting.read_datasets(common_path,binary_path)
	acc=voting.cross_acc(datasets)
	datasets=[ datasets[i] for i in np.argsort(acc)]
	results=[ train.train_model(data_i,binary)
				for data_i in datasets]
	acc=[voting.voting(results[:k+1]).get_acc() 
			for k in range(len(results))]
	return acc

def show_curve(acc,name="acc_curve",out_path=None):
    if(not name):
        name="acc_curve"
    plt.title(name)
    plt.grid(True)
    plt.xlabel('number of classifiers')
    plt.ylabel('accuracy')
    plt.plot(range(1,len(acc)+1), acc, color='red')
    if(out_path):
        plt.savefig(out_path)
    else:    
        plt.show()
    plt.clf()
    return acc

if __name__ == "__main__":
	common_path=['corl/person',
             'max_z/person']
#             '../agum/skew/person']
	acc=acc_curve(common_path,'../3DHOI/ens/feats',binary=False)
	show_curve(acc,"3DHOI")
	print(acc)
	