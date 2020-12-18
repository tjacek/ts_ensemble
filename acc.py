import numpy as np
import matplotlib.pyplot as plt
import voting,train

def acc_exp(common_path,binary_path,binary,out_path=None):
    acc=acc_curve(common_path,binary_path,binary)
    print(acc)
    if(out_path):
        np.savetxt(out_path,acc)
    show_curve(acc,name="acc_curve",out_path=None)

def acc_curve(common_path,binary_path,binary=False,acc_only=True):
    datasets=voting.read_datasets(common_path,binary_path)
    acc=voting.cross_acc(datasets)
    datasets=[ datasets[i] for i in np.argsort(acc)]
    results=[ train.train_model(data_i,binary)
                for data_i in datasets]
    acc=[voting.voting(results[:k+1])
    for k in range(len(results))]
    if(acc_only):
        acc=[ result_i.get_acc() for result_i in acc]
    return acc

def selected_clf(common_path,binary_path,binary,k,out_path):
    result=acc_curve(common_path,binary_path,binary,acc_only=False)
    print(result[k].get_acc())
    print(result[k].get_cf(out_path))

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
    dataset="MHAD"
    common_path=['../%s/dtw/max_z/person' % dataset,
                '../%s/dtw/corl/person' % dataset]
    binary_path='../%s/ens/feats' % dataset
#    acc_exp(common_path,binary_path,False)
    k=11
    out_path="cf%s" % dataset
    selected_clf(common_path,binary_path,False,k,out_path)