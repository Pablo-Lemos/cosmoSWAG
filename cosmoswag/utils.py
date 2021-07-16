import torch

def make_triangular(x, size):
    assert size*(size+1)//2 == x.shape[1], 'Wrong size'
    mat = torch.zeros([x.shape[0], size, size])
    j = 0
    n = size
    for i in range(size):
        mat[:,i, i:] = x[:,j:j+n]
        j+=n
        n-=1
    return mat