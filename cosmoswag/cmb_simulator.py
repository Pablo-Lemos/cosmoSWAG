"""
This file generates CMB simulations for a given cosmology.

It also contains functions to use on the CLs, such as binning, normalizing...
"""

import os
import numpy as np
import camb
import torch
from data_object import CMBDataObject


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


if __name__ == '__main__':
    N = 100000
    cls, params = generate_sims(N=N, save=True)
    assert (cls.shape == (N, 2549))
    assert (params.shape == (N, 5))
    print('COMPLETED')
