import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import files

def tsne_plot(in_path,show=True,color_helper="cat",names=False):
    feat_dataset= files.get_feats(in_path)
    tsne=manifold.TSNE(n_components=2,perplexity=30)#init='pca', random_state=0)
    X,y=feat_dataset.as_dataset()
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

tsne_plot('max_z/dtw')