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
from .utils import soft_clamp, make_triangular, logsumexp

class SWAGModel(nn.Module):

    def __init__(self, nin, npars, ncomps=1, cov_type="diag", device=None):
        #super(self.__class__, self).__init__()
        nn.Module.__init__(self)

        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.n_models = 0
        self.K = 20
        self.c = 2
        self.current_epoch = 1
        self.opt = None
        self.nin = nin
        self.npars = npars # The number of parameters
        self.ncomps = ncomps
        self.cov_type = cov_type
        if cov_type == "diag":
            self.nout = int((2*self.npars + 1)*self.ncomps)
        elif cov_type == "full":
            self.nout = int((self.npars + self.npars * (self.npars + 1)//2 + 1) * self.ncomps)
        else:
            print("Covariance type not known")
            raise

        if device is None:
            self._device = torch.device(
                'cuda:0' if torch.cuda.is_available() else 'cpu')
        else: 
            self._device = device

    def get_device(self):
        return self._device

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

    def load_weights(self, p_vec):
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

            #sigma = torch.abs(torch.diag(avg_w2 - avg_w ** 2))
            #w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 @ sigma ** 0.5
            w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 * torch.abs(
                avg_w2 - avg_w ** 2) ** 0.5
            w += scale * (D @ z_2).T / np.sqrt(2 * (K - 1))
            w = w[0]

        self.load_weights(w)
        return w

    def forward_swag(self, x, scale=0.5):
        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)
        return self.forward(x)

    def add_noise(self, x, delta_x = None, cov_x = None):
        # Add noise to the input
        if delta_x is not None:
            xs = x + torch.normal(0, delta_x) 
        elif cov_x is not None:
            m = torch.distributions.multivariate_normal \
                .MultivariateNormal(torch.zeros(cov_x.shape[0]),
                                    covariance_matrix=cov_x)
            xs = x + m.sample()
        else:
            xs = x
        return xs

    def generate_samples(self, x, nsamples, delta_x = None, cov_x = None, scale=0.5, verbose=True):
        samples = torch.zeros([nsamples, x.shape[0], self.nout])
        for i in range(nsamples):
            if (i % 100) == 0 and (i > 0) and (verbose):
                print(f"Generated {i} samples.")
            xs = self.add_noise(x, delta_x, cov_x)
            samples[i] = self.forward_swag(xs, scale=scale)
        return samples

    def sample_gmm(self, x, nsamples, delta_x = None, cov_x = None):
        # Sample from a GMM
        assert len(x.shape) == 2, "Input must be a 2D tensor"
        assert x.shape[1] == self.nin, "Wrong input size"
        x = self.add_noise(x, delta_x, cov_x)
        mu, invcov, alphas = self.separate_gmm(self(x))
        samples = torch.zeros([nsamples, x.shape[0], self.nout])
        r = torch.rand([nsamples, x.shape[0], 1])
        for i in range(self.ncomps):
            m = torch.distributions.multivariate_normal.MultivariateNormal(
                mu[:,i], precision_matrix = invcov[:,i])
            b = ((r > 0) * (r < alphas[:,i]))
            r = r - alphas[:,i].reshape([1, -1, 1])
            samples = samples + b*m.sample([nsamples])
        return samples

    def separate_mu_cov(self, pred):
        mu = pred[:, :self.npars]
        errors = pred[:, self.npars:]
        c = make_triangular(errors, self.npars, self._device)
        invcov = torch.einsum('...ij, ...kj -> ... ik', c, c)
        return mu, invcov

    def separate_mu_sigma(self, pred):
        mu = pred[:, :self.npars]
        log_sigma_squared = soft_clamp(pred[:, self.npars:], -50, 50)
        return mu, log_sigma_squared

    def separate_gmm(self, pred):
        mu = pred[:, :int(self.npars*self.ncomps)]
        ilow = int(self.npars * self.ncomps)
        if self.cov_type == "diag":
            ihigh = ilow + int(self.npars*self.ncomps)
            log_sigma_squared = pred[:, ilow:ihigh]
            log_sigma_squared = soft_clamp(log_sigma_squared, -50, 50)
            sigma = torch.reshape(log_sigma_squared, [-1, self.ncomps,
                                                     self.npars])
        else:
            ihigh = ilow + int(self.npars*(self.npars + 1)//2 * self.ncomps)
            errors = pred[:, ilow:ihigh]
            errors = torch.reshape(errors, [-1, self.npars*(self.npars + 1)//2])
            c = make_triangular(errors, self.npars, self._device)
            invcov = torch.einsum('...ij, ...kj -> ... ik', c, c)
            sigma = torch.reshape(invcov, [-1, self.ncomps, self.npars,
                                            self.npars])

        log_alphas = pred[:, ihigh:]

        # Shape: Data, comps, pars
        mu = torch.reshape(mu, [-1, self.ncomps, self.npars])
        # Normalize alphas
        alphas = torch.exp(log_alphas)
        alphas = alphas/torch.sum(alphas, keepdim=True, dim=1)
        alphas = torch.reshape(alphas, (-1, self.ncomps, 1))
        # sigma is log_sigma_squared for diagonal cov, invcov for full
        return mu, sigma, alphas

    def predict(self, x):
        pred = self(x)
        mu, invcov = self.separate_mu_cov(pred)
        return mu, invcov

    def get_logp_gmm(self, mu, y, sigma, alphas):
        if self.cov_type == "diag":
            loss = (mu - y) ** 2 / 2 / torch.exp(sigma) + sigma / 2
        else:
            alphas = alphas[:,:,0]
            chi2 = torch.einsum('...j, ...jk, ...k -> ...', mu - y,
                                sigma, mu - y)
            loss = chi2 / 2 - 0.5 * torch.log(torch.clip(torch.det(sigma),
                                                         min=1e-25, max=1e25))

        arg = loss + torch.log(alphas)
        loss = torch.logsumexp(arg, dim=1, keepdim=False)
        return loss

    def train(self, x_train, y_train, delta_x=None, cov_x=None, lr=1e-3,
                batch_size=32, num_workers=6, num_epochs=10000,
              pretrain=False, mom_freq=100, patience=20, save_every=0,
              save_name=None, save_path=None):
        """Train the model"""

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []

        if delta_x is not None:
            delta_x = delta_x.to(self._device)
        elif cov_x is not None:
            cov_x = cov_x.to(self._device)
            m = torch.distributions.multivariate_normal \
                .MultivariateNormal(torch.zeros(cov_x.shape[0], device=self._device),
                                    covariance_matrix=cov_x)

        dataset = data_utils.TensorDataset(x_train, y_train)
        trainloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        count = 0

        if pretrain:
            num_steps_no_improv = 0
            best_loss = np.infty

        t = trange(num_epochs, desc='loss', leave=True)
        for i in t:
            losses = []
            for x, y in trainloader:
                self.opt.zero_grad()
                inp = x.to(self._device)
                out = y.to(self._device)
                if delta_x is not None:
                    delta = torch.normal(0, delta_x)
                    inp = inp + delta
                elif cov_x is not None:
                    delta = m.sample()
                    inp = inp + delta

                pred = self(inp)
                mu, sigma, alphas = self.separate_gmm(pred)
                alphas = torch.reshape(alphas, (-1, self.ncomps, 1))
                y = torch.reshape(out, (-1, 1, self.npars))

                loss = self.get_logp_gmm(mu, y, sigma, alphas)
                loss = loss.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                #nn.utils.clip_grad_value_(self.parameters(), 10.0)
                self.opt.step()
                count += 1
                losses.append(loss.item())

            if ((save_every > 0) and (i%save_every == 0)): 
                self.save(name=save_name, path=save_path) 

            t.set_description(f"Loss = {np.average(losses) :.5f}", refresh=True)
            self.current_epoch = i

            if pretrain:
                if np.average(losses) < best_loss:
                    best_loss = np.average(losses)
                    num_steps_no_improv = 0
                    best_state_dict = self.state_dict()
                elif np.isfinite(np.average(losses)):
                    num_steps_no_improv += 1

                if (num_steps_no_improv > patience):
                    print("Early stopping after ", num_steps_no_improv,
                          "epochs, and", count, "steps.")
                    self.load_state_dict(best_state_dict)
                    return None
            else:
                self.aggregate_model()

    def save(self, name=None, path=None):
        """ Save the model"""
        if not name:
            name = 'swag.pt'
            print("No name provided, using default: " + name)

        if path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'data/saved_models/')
            print("No path provided, using default: " + dir_path)
        
        path = os.path.join(path, name)

        torch.save([self.state_dict(), self.opt.state_dict(),
                    self.current_epoch, self.w_avg,
                    self.w2_avg, self.pre_D], path)

    def load(self, name, path=None):
        if path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            path = os.path.join(dir_path, 'data/saved_models/')
            print("No path provided, using default: " + dir_path)

        path = os.path.join(path, name)

        if not os.path.isfile(path):
            print("File does not exist: " + path)
            return None 

        try:
            model_state_dict, opt_state_dict, current_epoch ,self.w_avg, self.w2_avg, self.pre_D = torch.load(path)
        except:
            model_state_dict, opt_state_dict, current_epoch ,self.w_avg, \
            self.w2_avg, self.pre_D = torch.load(path, map_location=torch.device('cpu'))

        self.load_state_dict(model_state_dict)
        if self.opt is None:
            self.opt = torch.optim.Adam(self.parameters())
        self.opt.load_state_dict(opt_state_dict)


