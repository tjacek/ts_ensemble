import numpy as np
import matplotlib.pyplot as plt
import voting,train

def acc_exp(common_path,binary_path,binary,out_path):
    acc=acc_curve(common_path,binary_path,binary)
    print(acc)
    print(acc[7])
    if(out_path):
        np.savetxt(out_path,acc)
    show_curve(acc,name="acc_curve",out_path=None)

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
	common_path=['../3DHOI/dtw/max_z/person',
             '../3DHOI/dtw/corl/person']
#             '../agum/skew/person']
	acc_exp(common_path,'../3DHOI/ens/feats',False,'acc/3DHOI')
#	print(acc)
	