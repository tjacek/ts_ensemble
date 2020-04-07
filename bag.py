import random
import files

def make_bagset(in_path,out_path,k=7):
    seq_dict=files.get_seqs(in_path)
    files.make_dir(out_path)
    for i in range(k):
        dataset_i=resample(seq_dict)
        out_i="%s/bag%d" % (out_path,i)
        files.save_seqs(dataset_i,out_i)
    files.save_seqs(seq_dict,"%s/full"%out_path)

def resample(seq_dict):
    train,test=files.split(seq_dict)
    names=list(train.keys())
    sampled_names=random.choices(names,k=len(names))
    bag_dict={ ("%s_%d" % (name_i,i)):train[name_i]  
                for i,name_i in enumerate(sampled_names)}
    bag_dict.update(test)
    return bag_dict

make_bagset("agum","bagging",k=7)