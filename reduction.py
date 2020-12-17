import os,numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import feats#files

def rec_tsne(in_path,out_path):
    for root, dirs, content in os.walk(in_path):
        out_dir=root.replace(in_path,out_path)
        feats.make_dir(out_dir)
        if(content):
            for content_i in content:
                in_i="%s/%s" % (root,content_i)
                out_i="%s/%s" % (out_dir,content_i)
                print(out_i)
                plot_i= tsne_plot(in_i,show=False)
                plot_i.savefig(out_i,dpi=1000)
                plot_i.close()

def tsne_plot(in_path,show=True,color_helper="cat",names=False):
    feat_dataset= feats.read_feats(in_path).split()[1]  #files.get_feats(in_path)
    tsne=manifold.TSNE(n_components=2,perplexity=30)#init='pca', random_state=0)
    X,y=feat_dataset.to_dataset()
    X=tsne.fit_transform(X)
    names=feat_dataset.info if(names) else None
    color_helper=lambda i,y_i:y_i 
    return plot_embedding(X,y,title="tsne",color_helper=color_helper,show=show,names=names)

def plot_embedding(X,y,title="plot",color_helper=None,show=True,names=None):
    n_points=X.shape[0]
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
   
    color_helper=color_helper if(color_helper) else lambda i,y_i:0

    plt.figure()
    ax = plt.subplot(111)

    rep= names if(names) else y
    for i in range(n_points):
        color_i= color_helper(i,y[i])
        plt.text(X[i, 0], X[i, 1],str(rep[i]),
                   color=plt.cm.tab20( color_i),
                   fontdict={'weight': 'bold', 'size': 9})
    print(x_min,x_max)
    if title is not None:
        plt.title(title)
    if(show):
        plt.show()
    return plt

#tsne_plot('max_z/dtw')
rec_tsne('../MHAD/ens/feats',"../MHAD/ens/plots")