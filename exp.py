import preproc,frame,files
import spline,block,bag

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

#exp("../MSR/box","simple")
smooth_exp("../MSR/box","gauss")