import torch
from torch import nn
import torchvision
import numpy as np
from collections import OrderedDict


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
        # print(x.shape)
        return cropped


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        # print(x.shape)
        return x


class MyModel_try_2(nn.Module):

    def __init__(self, hidden=8, kernel=3):
        super(self.__class__, self).__init__()

        self.cropper = RandomCrop()
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.n_models = 0
        self.K = 20
        self.c = 2
        self.current_epoch = 1

        self.flipper = nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5)
        )

        self.conv = nn.Sequential(
            nn.Conv3d(1, hidden, kernel, dilation=1),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=1),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=1),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=2),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=2),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=2),
            nn.ReLU(),
            nn.Conv3d(hidden, hidden, kernel, dilation=2)

        )
        crop = 24
        example_data = torch.zeros(1, 1, crop, crop, crop)
        example_out = self.conv(example_data).mean((2, 3, 4))

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(example_out.shape[1], 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 10)  # will change 1 -> 5 for all parameters # change 5 -> 1 for predicting 1 parameter
        )

        self.default_crop_size = 24

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def forward(self, x, crop_size=None):
        if crop_size is None:
            crop_size = self.default_crop_size

        x = self.cropper(x, (crop_size, crop_size, crop_size)) * 1000
        x = self.flipper(x)
        # print(x.shape)
        x2 = self.conv(x)
        # print(x2.shape)
        x3 = x2.mean((2, 3, 4))

        return self.out(x3)

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
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((K, 1), device=self.device)
            sigma = torch.abs(torch.diag(avg_w2 - avg_w ** 2))

            w = avg_w[None] + scale * (1.0 / np.sqrt(2.0)) * z_1 @ sigma ** 0.5
            w += scale * (D @ z_2).T / np.sqrt(2 * (K - 1))
            w = w[0]

        self.load(w)

    def forward_swag(self, x, scale=0.5):
        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)
        return self.forward(x)


----------------------------------------------------------------------------------------

import torch
from torch import nn
from importlib import reload
import cnn_redo_v1

reload(cnn_redo_v1)
from cnn_redo_v1 import MyModel_try_2
import numpy as np
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader, random_split
import torch.utils.data as data_utils
import os


# from get_files import train_inp_torch, train_out_torch

def soft_clamp(x, low, high):
    range = (high - low)
    return torch.sigmoid(x) * range + low


def train_data(num_epochs, mod_num, train_inp_torch, train_out_torch):
    losses = []
    mod = MyModel_try_2(hidden=10).cuda()
    opt = torch.optim.Adam(mod.parameters(), lr=1e-3)
    indices = np.arange(0, 1800, 1)
    ind_2 = np.arange(0, 8, 1)
    losses = []

    dataset = data_utils.TensorDataset(train_inp_torch, train_out_torch)
    trainloader = DataLoader(dataset, batch_size=8, num_workers=6, shuffle=True)
    count = 0

    for i in trange(num_epochs):
        #         shuf = torch.tensor(shuffle(indices))
        #         opt.zero_grad()
        #         y = torch.index_select(train_out_torch, 0, shuf)[ind_2]
        #         inp = torch.index_select(train_inp_torch, 0, shuf)[ind_2]

        #         y = torch.randn(50, 1)
        #         inp = torch.ones(50, 1, 128, 128, 128) * y[:, :, None, None, None]
        #        y = y * torch.ones(50, 5)
        # print(y.shape)
        # print(inp.shape)

        # inp += torch.sin(torch.randn(10, 1)[:, :, None, None, None]*10 +
        #                  y[:, :, None, None, None] * torch.arange(64)[None, None, :, None, None]/16*np.pi)
        # print(model.device)
        for x, y in trainloader:
            opt.zero_grad()
            inp = x.cuda()
            y = y.cuda()
            pred = mod(inp)
            # loss = F.mse_loss(pred[:, 0::2], y)
            mu = pred[:, 0::2]
            log_sigma_squared = soft_clamp(pred[:, 1::2], -4, 4)

            print("y")
            print(y)
            print("mu")
            print(mu)
            print("sigma")
            print(log_sigma_squared)
            loss = (mu - y) ** 2 / 2 / torch.exp(log_sigma_squared) + log_sigma_squared / 2

            # loss = (mu - y)**2
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(mod.parameters(), 1.0)
            opt.step()
            count += 1
            losses.append(loss.item())
            if count % 50 == 0 and count > 0:
                print(np.average(losses))
                # print(mu-y)
                losses = []
            if count % 5 == 0 and i == num_epochs - 1:
                mod.aggregate_model()
                # print(model.pre_D)mode

    os.chdir('/scratch/gpfs/sslav/Thesis')
    save_name = 'mod_test_' + str(mod_num) + '.pt'
    # torch.save(model.state_dict(), save_name)
    torch.save([mod.state_dict(), mod.w_avg, mod.w2_avg, mod.pre_D], save_name)