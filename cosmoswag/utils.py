import torch


def make_triangular(x, size, device = None):
    assert size * (size + 1) // 2 == x.shape[1], 'Wrong size'
    if device is not None:
        mat = torch.zeros([x.shape[0], size, size], device=device, dtype=torch.float64)
    else:
        mat = torch.zeros([x.shape[0], size, size], dtype=torch.float64)
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
            x = torch.tensor(x, dtype=torch.float64)
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


def logsumexp(a, dim=None, b=None, keepdim=False, return_sign=False):
    if b is not None:
        a, b = torch.broadcast_tensors(a, b)
        if torch.any(b == 0):
            a = a + 0.  # promote to at least float
            a[b == 0] = -torch.inf

    a_max = torch.amax(a, dim=dim, keepdim=True)
    a_maxx = torch.max(a, dim=dim, keepdim=True)

    # if a_max.ndim > 0:
    #     a_max[~torch.isfinite(a_max)] = 0
    # elif not torch.isfinite(a_max):
    #     a_max = 0

    if b is not None:
        b = torch.as_tensor(b)
        tmp = b * torch.exp(a - a_max)
    else:
        tmp = torch.exp(a - a_max)

    # suppress warnings about log of zero
    #with torch.errstate(divide='ignore'):
    s = torch.sum(tmp, dim=dim, keepdim=keepdim)
    if return_sign:
        sgn = torch.sign(s)
        s = s * sgn  # /= makes more sense but we need zero -> zero
    out = torch.log(s)

    if not keepdim:
        a_max = torch.squeeze(a_max, dim=dim)
    out = out + a_max

    if return_sign:
        return out, sgn
    else:
        return out

if __name__ == "__main__":
    a = torch.randn(3, 3)
    print(logsumexp(a, b = torch.randn(3), dim=1))
