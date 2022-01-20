"""
This file generates CMB simulations for a given cosmology.

It also contains functions to use on the CLs, such as binning, normalizing...
"""

import os
import numpy as np
import camb
import torch


def generate_camb_cl(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06, logA=2.5, ns=0.965, r=0):
    """ Generate a temperature power spectrum given cosmological parameters"""
    As = 1e-10 * np.exp(logA)
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns, r=r)
    pars.set_for_lmax(2500, lens_potential_accuracy=0);
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    totCL = powers['total']
    return totCL[2:, 0]


def bin_cls(cls, nl):
    """ Return a binned version of the CLs

    N is approximately the number of elements left (although in reality it will be less than nl)
    """
    binned_l = np.unique(np.logspace(0, np.log10(cls.shape[1]), nl, dtype='int'))-1
    binned_cls = cls[:, binned_l]
    return binned_l, binned_cls

def bin_cls_to(cls, bin_centers):
    """ Return a binned version of the CLs for given bins
    """
    edges = bin_centers[1:] - (bin_centers[1:] - bin_centers[:-1])/2
    binned = np.zeros([cls.shape[0], len(bin_centers)])
    for i, _ in enumerate(edges):
        if i==0:
            binned[:,i] = np.mean(cls[:,:int(edges[0])], axis=1)
        elif i==len(edges)-1:
            binned[:,i] = np.mean(cls[:,int(edges[-1]):], axis=1)
        else:
            binned[:,i] = np.mean(cls[:,int(edges[i-1]):int(edges[i])], axis=1)

    return binned

def normalize_params(theta, mins=None, maxs=None):
    """ Normalize the parameters"""
    if maxs is None:
        maxs = np.array([90, 0.05, 0.5, 3.5, 1])
    if mins is None:
        mins = np.array([50, 0.01, 0.01, 1.5, 0.8])

    return (theta - mins) / (maxs - mins)

def normalize_cls(cls):
    """ Normalize the data"""
    return (cls - np.mean(cls, axis=0, keepdims=True)) / np.sqrt(np.var(
        cls, axis=0, keepdims=True))

def unnormalize_params(theta, mins=None, maxs=None):
    """ Undo parameter normalization"""
    if maxs is None:
        maxs = np.array([90, 0.05, 0.5, 3.5, 1])
    if mins is None:
        mins = np.array([50, 0.01, 0.01, 1.5, 0.8])

    return theta * (maxs - mins) + mins

def compute_derivatives(theta_fiducial, nl, step, normed = True):
    """ Calculate derivatives with respect to fiducial parameters (used for MOPDED compression)"""

    npars = len(theta_fiducial)

    if normed:
        theta_fiducial = unnormalize_params(theta_fiducial)

    # Holder for derivatives
    dCdt = np.zeros([npars,nl])

    # Derivatives wrt cosmological parameters
    for i in range(npars):
        theta_plus = np.copy(theta_fiducial)
        theta_plus[i] += step[i]
        theta_minus = np.copy(theta_fiducial)
        theta_minus[i] -= step[i]

        Cp = generate_y(theta_plus)
        Cm = generate_y(theta_minus)

        dCdt[i,:] = (Cp - Cm)/(2*step[i])

    return dCdt



def generate_sims(N, mins=None, maxs=None, save=True, path=None):
    """ Generated N randomly generates CLs """

    if maxs is None:
        maxs = np.array([90, 0.05, 0.5, 3.5, 1])
    if mins is None:
        mins = np.array([50, 0.01, 0.01, 1.5, 0.8])

    cls = np.zeros([N, 2549])
    H0_rand = np.random.uniform(mins[0], maxs[0], N)
    ombh2_rand = np.random.uniform(mins[1], maxs[1], N)
    omch2_rand = np.random.uniform(mins[2], maxs[2], N)
    logA_rand = np.random.uniform(mins[3], maxs[3], N)
    ns_rand = np.random.uniform(mins[4], maxs[4], N)

    skipped = []
    for i in range(N):
        if i % 500 == 0: print('Generated ', i, 'out of ', N)

        try:
            cls[i] = generate_camb_cl(H0=H0_rand[i],
                                      ombh2=ombh2_rand[i],
                                      omch2=omch2_rand[i],
                                      logA=logA_rand[i],
                                      ns=ns_rand[i])
        except:
            print("Skipped iteration ", i)
            skipped.append(i)

    params = np.stack([H0_rand,
                       ombh2_rand,
                       omch2_rand,
                       logA_rand,
                       ns_rand]).T

    params = np.delete(params, skipped, axis=0)

    if save:
        if not path:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'data/cmb_sims/')

        np.save(os.path.join(path, 'cls'), cls)
        np.save(os.path.join(path, 'params'), params)

    return cls, params

def read_data(path = None, binned = False, normalize = True):
    if not path:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, 'data/')

    cls = np.load(os.path.join(path, 'cmb_sims/cls.npy'))
    params = np.load(os.path.join(path, 'cmb_sims/params.npy'))

    truth_filename = "planck/COM_PowerSpect_CMB-TT-full_R3.01.txt" if binned \
        else "planck/COM_PowerSpect_CMB-TT-full_R3.01.txt"
    truth = np.loadtxt(os.path.join(path, truth_filename))

    theta_norm = normalize_params(params)

    if binned:
        # Use only high ell
        cls = cls[:, 29:]
        cls = bin_cls_to(cls, truth[:, 0])
        #cls[-1] = truth[:, 1]
        # Last point is not calculated
        cls = cls[:, :-1]
        truth = truth[:-1]
    else:
        # Match to cls from data
        cls = cls[:, :truth.shape[0]]

    # Normalize
    if normalize: cls = normalize_cls(cls)

    error = truth[:, 2]

    x_train = torch.tensor(cls[:-1000], dtype=torch.float32)
    y_train = torch.tensor(theta_norm[:-1000], dtype=torch.float32)
    x_val = torch.tensor(cls[-1000:], dtype=torch.float32)
    y_val = torch.tensor(theta_norm[-1000:], dtype=torch.float32)
    delta_y = torch.tensor(error, dtype=torch.float32)

    return x_train, y_train, x_val, y_val, delta_y

if __name__ == '__main__':
    N = 100000
    cls, params = generate_sims(N=N, save=True)
    assert (cls.shape == (N, 2549))
    assert (params.shape == (N, 5))
    print('COMPLETED')
