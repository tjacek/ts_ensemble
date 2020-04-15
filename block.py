import numpy as np
import files,agum

def make_blocks(in_path,out_path,k=8,t=8):
    def block_helper(seq_i):
        return [sample_blocks(seq_i,k) for j in range(t)]
    agum.agum_template(in_path,out_path,[block_helper])

def sample_blocks(seq_i,k):
    n_blocks=int(seq_i.shape[0]/k)
    max_j=seq_i.shape[0]-k
    indexes=np.random.randint(max_j, size=n_blocks)
    indexes=np.sort(indexes)
    blocks=[seq_i[j:j+k]  for j in indexes]
    new_seq_j=np.concatenate(blocks,axis=0)
    return new_seq_j

make_blocks('../MSR1/spline','../MSR2/block1')