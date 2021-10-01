""" Use multiSWAG to get parameter constrains simulations containing array data.

Based on code by Miles Cranmer and Shai Slav"""

import torch
from torch import nn
import numpy as np
from collections import OrderedDict
from tqdm.auto import trange
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import os
from utils import make_triangular

class RandomCrop(nn.Module):
    def __init__(self):
        super(self.__class__, self).__init__()

    def forward(self, x, shape):
        # print(x.shape)
        shape = torch.tensor(list(shape)).to(device=x.device)
        input_shape = torch.tensor(x.shape[2:], device=x.device)
        start_x = torch.tensor([torch.randint(
            low=0,
            high=input_shape[i] - shape[i],
            size=()) for i in range(3)]).to(device=x.device)
        end_x = start_x + shape
        cropped = x[:, :,
                  start_x[0]:end_x[0],
                  start_x[1]:end_x[1],
                  start_x[2]:end_x[2]]
        return cropped


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        return x


class SWAGModel(nn.Module):

    def __init__(self, nin, npars):
        super(self.__class__, self).__init__()

        self.cropper = RandomCrop()
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.n_models = 0
        self.K = 20
        self.c = 2
        self.current_epoch = 1
        self.npars = npars # The number of parameters
        nout = self.npars + self.npars * (self.npars + 1) // 2 # We output parameters and a covariance matrix

        hidden = 64

        layers = (
            [torch.nn.LayerNorm(nin)]
            + [torch.nn.Linear(nin, hidden), torch.nn.ReLU()]
            + [[torch.nn.Linear(hidden, hidden), torch.nn.ReLU()][i%2] for i in range(2*2)]
            + [torch.nn.Linear(hidden, nout)]
        )
        self.out = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)

    def aggregate_model(self):
        # """Aggregate parameters for SWA/SWAG"""

        cur_w = self.flatten()
        cur_w2 = cur_w ** 2
        with torch.no_grad():
            if self.w_avg is None:
                self.w_avg = cur_w
                self.w2_avg = cur_w2
            else:
                self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
                self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

            if self.pre_D is None:
                self.pre_D = cur_w.clone()[:, None]
            else:
                # Record weights, measure discrepancy with average later
                self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
                if self.pre_D.shape[1] > self.K:
                    self.pre_D = self.pre_D[:, 1:]

        self.n_models += 1
        print("num agg = " + str(self.n_models))

    def flatten(self):
        # """Convert state dict into a vector"""
        ps = self.state_dict()
        p_vec = None
        for key in ps.keys():
            p = ps[key]
            if p_vec is None:
                p_vec = p.reshape(-1)
            else:
                p_vec = torch.cat((p_vec, p.reshape(-1)))
        return p_vec

    def load(self, p_vec):
        # """Load a vector into the state dict"""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key in cur_state_dict.keys():
            old_p = cur_state_dict[key]
            size = old_p.numel()
            shape = old_p.shape
            new_p = p_vec[i:i + size].reshape(*shape)
            new_state_dict[key] = new_p
            i += size

        self.load_state_dict(new_state_dict)

    def sample_weights(self, scale=1):
        #         """Sample weights using SWAG:
        #         - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
        #             - This can be done with the following matrices:
        #                 - z_1 ~ N(0, I_d); d the number of parameters
        #                 - z_2 ~ N(0, I_K)
        #             - Then, compute:
        #             - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        #         """
        with torch.no_grad():
            avg_w = self.w_avg  # [K]
            avg_w2 = self.w2_avg  # [K]
            D = self.pre_D - avg_w[:, None]  # [d, K]
            d = avg_w.shape[0]
            K = self.K
            z_1 = torch.randn((1, d))#, device=self.device)
            z_2 = torch.randn((K, 1))#, device=self.device)

            sigma = torch.abs(torch.diag(avg_w2 - avg_w ** 2))

            w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 @ sigma ** 0.5
            w += scale * (D @ z_2).T / np.sqrt(2 * (K - 1))
            w = w[0]

        self.load(w)
        return w

    def forward_swag(self, x, scale=0.5):
        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)
        return self.forward(x)

    def separate_mu_cov(self, pred):
        mu = pred[:, :self.npars]
        errors = pred[:, self.npars:]
        c = make_triangular(errors, self.npars)
        invcov = torch.einsum('...ij, ...kj -> ... ik', c, c)
        return mu, invcov

    def predict(self, x):
        pred = self(x)
        mu, invcov = self.separate_mu_cov(pred)
        return mu, invcov

    def train(self, x_train, y_train, x_val, y_val, lr=1e-3, batch_size=32, num_workers=6, num_epochs=10000):
        """Train the model"""

        #TODO Add early stopping
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        losses = []

        dataset = data_utils.TensorDataset(x_train, y_train)
        trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        count = 0

        for i in trange(num_epochs):
            for x, y in trainloader:
                opt.zero_grad()
                inp = x  # .cuda()
                y = y  # .cuda()
                pred = self(inp)
                mu, invcov = self.separate_mu_cov(pred)
                chi2 = torch.einsum('...j, ...jk, ...k -> ...', mu - y, invcov, mu - y)
                loss = chi2 / 2 - 0.5 * torch.log(torch.clip(torch.det(invcov), min=1e-25, max=1e25))

                loss = loss.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                count += 1
                losses.append(loss.item())
                if count % 1000 == 0 and count > 0:
                    print(np.average(losses))
                    losses = []
                if count % 5 == 0 and i == num_epochs - 1:
                    self.aggregate_model()

    def save(self, path=None, name=None):
        """ Save the model"""
        # TODO Miles, how can I load a saved model?
        if not name:
            name = 'swag'

        if not path:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'data/saved_models/', name)

        torch.save([self.state_dict(), self.w_avg, self.w2_avg, self.pre_D], path)

