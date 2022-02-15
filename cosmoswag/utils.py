import torch


def make_triangular(x, size):
    assert size * (size + 1) // 2 == x.shape[1], 'Wrong size'
    mat = torch.zeros([x.shape[0], size, size])
    j = 0
    n = size
    for i in range(size):
        mat[:, i, i:] = x[:, j:j + n]
        j += n
        n -= 1
    return mat


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    else:
        try:
            x = torch.tensor(x, dtype=torch.float32)
        except:
            raise TypeError('Could not convert to tensor.')
        return x


def soft_clamp(x, low, high):
    range = (high - low)
    return torch.sigmoid(x) * range + low


def cov3d(x):
    N = x.shape[0]
    m1 = x - torch.sum(x, dim=0, keepdim = True)/N
    out = torch.einsum('kij,kil->ijl',m1,m1)  / (N - 1)
    return out