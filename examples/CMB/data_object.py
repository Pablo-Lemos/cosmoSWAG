import os
import torch
import numpy as np
from cosmoswag import to_tensor


class DataObject(object):
    def __init__(self, data, params, validation_split=0.1, norm_data=True,
                 norm_params = True):
        data, params = to_tensor(data), to_tensor(params)
        assert len(data.shape) == len(params.shape) == 2, 'Data or parameters ' \
                                                          'are not two ' \
                                                          'dimensional '
        assert data.shape[0] == params.shape[0], 'Different number of data ' \
                                                 'and params '
        self.nSims = data.shape[0]
        self.dataSize = data.shape[1]
        self.parameterSize = params.shape[1]

        # Data normalization attributes
        self.dataMean = None
        self.dataStd = None
        self.xTrain, self.yTrain, self.xVal, self.yVal = self.split(data, params,
                                                                  validation_split)
        if norm_data:
            self.xTrain = self.normalize_data(self.xTrain)
            self.xVal = self.normalize_data(self.xVal)

        self.paramMins, _ = torch.min(self.yTrain, dim=0)
        self.paramMaxs, _ = torch.max(self.yTrain, dim=0)

        if norm_params:
            self.yTrain, self.yVal = self.normalize_params(self.yTrain), \
                                     self.normalize_params(self.yVal)

    def split(self, data, params, ratio):
        split_index = int(self.nSims * ratio)
        x_train = data[:-split_index]
        y_train = params[:-split_index]
        x_val = data[-split_index:]
        y_val = params[-split_index:]
        return x_train, y_train, x_val, y_val

    def normalize_data(self, data):
        """ Normalize the data"""
        assert len(data.shape) == 2, 'Data must be two dimensional'
        assert data.shape[1] == self.dataSize, 'Wrong data size'
        if self.dataMean is None:
            self.dataMean = torch.mean(self.xTrain, dim=0, keepdim=True)
            self.dataStd = torch.std(self.xTrain, dim=0, keepdim=True)
        return (data - self.dataMean) / self.dataStd

    def unnormalize_data(self, data):
        assert len(data.shape) == 2, 'Data must be two dimensional'
        assert data.shape[1] == self.dataSize, 'Wrong data size'
        if self.dataMean is None:
            print('DataObject has no normalization')
            return data

        return data * self.dataStd + self.dataMean

    def normalize_params(self, theta):
        assert len(theta.shape) == 2, 'Data must be two dimensional'
        assert theta.shape[1] == self.parameterSize, 'Wrong parameter size'
        return (theta - self.paramMins) / (self.paramMaxs - self.paramMins)

    def unnormalize_params(self, theta):
        assert len(theta.shape) == 2, 'Data must be two dimensional'
        assert theta.shape[1] == self.parameterSize, 'Wrong parameter size'
        return theta * (self.paramMaxs - self.paramMins) + self.paramMins

    def get_data(self):
        return self.xTrain, self.yTrain, self.xVal, self.yVal


class CMBDataObject(DataObject):
    def __init__(self, data, params, validation_split=0.1, norm_data=True):
        super().__init__(data, params, validation_split, norm_data)

    def normalize_error(self, error):
        """ Normalize the data"""
        assert len(error.shape) == 1, 'Data must be one dimensional'
        assert len(error) == self.dataSize, 'Wrong data size'
        return error if self.dataStd is None else error/self.dataStd

    def normalize_covariance(self, cov):
        """ Normalize the data"""
        assert len(cov.shape) == 2, 'Data must be two dimensional'
        assert cov.shape[0] == self.dataSize, 'Wrong data size'
        if self.dataStd is None:
            return cov
        else:
            d = np.outer(self.dataStd, self.dataStd)
            return cov/d

    def read_truth(self, path=None, binned=False,
                   filename="COM_PowerSpect_CMB-TT-full_R3.01.txt"):

        if path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'data/planck/')

        truth = np.loadtxt(os.path.join(path, filename))

        if binned:
            self.xTrain = bin_cls_to(self.xTrain, truth[:, 0])
            self.xVal = bin_cls_to(self.xVal, truth[:, 0])
            self.xTrain = self.xTrain[:, :-1]
            self.xVal = self.xVal[:, :-1]
            truth = self.truth[:-1]
        else:
            self.xTrain = self.xTrain[:, :truth.shape[0]]
            self.xVal = self.xVal[:, :truth.shape[0]]

        # Adjust data size
        self.dataSize = self.xTrain.shape[1]
        self.dataMean = self.dataMean[:, :self.dataSize]
        self.dataStd = self.dataStd[:, :self.dataSize]

        truth = to_tensor(truth.T)

        # Redo the normalization
        if self.dataMean is not None:
            self.truth = self.normalize_data(truth[1:2])

        self.error = self.normalize_error(truth[-1])
        self.ell = truth[0]

    def get_error(self):
        return self.error[0]

    def get_ell(self):
        return self.ell

    def get_truth(self):
        return self.truth

    def get_true_cls(self):
        return self.truth[1:2]


def bin_cls(cls, nl):
    """ Return a binned version of the CLs
    N is approximately the number of elements left (although in reality it will be less than nl)
    """
    binned_ell = np.unique(
        np.logspace(0, np.log10(cls.shape[1]), nl, dtype='int')) - 1
    binned_cls = cls[:, binned_ell]
    return binned_ell, binned_cls


def bin_cls_to(cls, bin_centers):
    """ Return a binned version of the CLs for given bins
    """
    cls = cls[:, 29:] # The binned truth only uses the high ells
    edges = bin_centers[1:] - (bin_centers[1:] - bin_centers[:-1]) / 2
    binned = np.zeros([cls.shape[0], len(bin_centers)])
    for i, _ in enumerate(edges):
        if i == 0:
            binned[:, i] = np.mean(cls[:, :int(edges[0])], axis=1)
        elif i == len(edges) - 1:
            binned[:, i] = np.mean(cls[:, int(edges[-1]):], axis=1)
        else:
            binned[:, i] = np.mean(cls[:, int(edges[i - 1]):int(edges[i])],
                                   axis=1)

        return binned


def read_data(path=None, normalize=True):
    if path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, 'data/cmb_sims')

    cls = np.load(os.path.join(path, 'cls.npy'))
    params = np.load(os.path.join(path, 'params.npy'))

    data = CMBDataObject(cls, params, norm_data=normalize)
    #data.read_truth()

    return data


def read_binned_data(path=None, normalize=True):
    if path is None:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(dir_path, 'data/cmb_sims')

    cls = np.load(os.path.join(path, 'cls_binned.npy'))
    params = np.load(os.path.join(path, 'params.npy'))

    data = CMBDataObject(cls, params, norm_data=normalize)
    #data.read_truth()

    return data


if __name__ == "__main__":
    data = read_data()
    data.read_truth()
    x_train, y_train, x_val, y_val = data.get_data()
    delta_x = data.get_error()
    truth = data.get_truth()
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape, delta_x.shape)
