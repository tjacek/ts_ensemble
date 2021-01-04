import preproc,frame,files
import spline,block,bag
import dtw

def dtw_exp(in_path,feat_type,out=False):
    files.make_dir(feat_type)
    seq_path="%s/seqs" % feat_type
    frame.compute(in_path,seq_path,
        feat_set=feat_type,out=out)
    dtw.make_dtw_feats(feat_type)

def ae_exp(in_path,out_path):
    files.make_dir(out_path)
    seq_dict=files.get_seqs(in_path)
    pairs=dtw.make_pairwise_distance(seq_dict)
    save(pairs, "%s/%s" % (out_path,"pairs"))

def unify_exp(paths,out_path,filename="dtw"):
    paths=["%s/%s" % (path_i,filename) 
            for path_i in paths]
    files.unify_feats(paths,out_path)


def scaled_exp(in_path,out_path):
    files.make_dir(out_path)
    scale_path=out_path+"/scale"
    preproc.scaled_frames(in_path,scale_path)
    exp(scale_path,out_path)

def smooth_exp(in_path,out_path):
    files.make_dir(out_path)
    scale_path=out_path+"/smooth"
    preproc.smooth_frames(in_path,scale_path)
    exp(scale_path,out_path)

def exp(in_path,out_path):
    files.make_dir(out_path)
    seq_path= out_path+"/seqs"
    frame.compute(in_path,seq_path)
    spline_path= out_path+"/spline"
    spline.upsample(seq_path,spline_path,size=128)
    block_path=out_path+"/blocks"
    block.make_blocks(spline_path,block_path,k=8,t=10)
    bag_path=out_path+"/bagging"
    bag.make_bagset(block_path,bag_path,k=7)
    feat_path=out_path+"/feats"
    bag.train_bag(bag_path,feat_path,n_epochs=1000)

if __name__ == '__main__':
#dtw_exp("box","skew",out=True)
#    unify_exp(["max_z","skew","corl"],"full2",filename="feats")
    ae_exp("../3DHOI_proj/agum/ae/seqs","../3DHOI_proj/agum/ae/dtw")