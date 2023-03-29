from cosmoswag import SWAGModel
import torch
from torch import nn
import torchvision
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import gc

Nside=32
seed=1


class CNNModel(SWAGModel):
    def __init__(self, n_pars, n_comps=1, cov_type="diag", kernel=3, Nside=128, nlayers_cnn=4, nhidden_cnn=4, nlayers_out=4, nhidden_out=128, dropout_cnn=0., dropout_out=0.1, device=None):
        #super(self.__class__, self).__init__()
        SWAGModel.__init__(self, 0, n_pars, n_comps, cov_type, device)
        
        self.flipper = nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5)
        )

        curr_hidden = nhidden_cnn

        self.conv = []
        self.conv.append(nn.Conv3d(1, curr_hidden, kernel, padding = 'same'))
        self.conv.append(nn.ReLU())
        self.conv.append(nn.MaxPool3d(2))
        self.conv.append(nn.BatchNorm3d(curr_hidden))
        self.conv.append(nn.Dropout(dropout_cnn))
        for i in range(nlayers_cnn - 1):
            self.conv.append(nn.Conv3d(curr_hidden, curr_hidden*2, kernel, padding = 'same'))
            curr_hidden*=2
            self.conv.append(nn.ReLU())
            self.conv.append(nn.MaxPool3d(2))
            self.conv.append(nn.BatchNorm3d(curr_hidden)),
            self.conv.append(nn.Dropout(dropout_cnn))

        self.conv = nn.Sequential(*self.conv)
        size = (Nside/(2**nlayers_cnn))**2*(Nside/2/(2**nlayers_cnn))*curr_hidden

        self.out = []
        self.out.append(nn.Flatten())
        self.out.append(nn.Linear(int(size), nhidden_out))
        self.out.append(nn.ReLU())
        self.out.append(nn.Dropout(dropout_out))

        for i in range(nlayers_out - 1):
            self.out.append(nn.Linear(nhidden_out, nhidden_out))
            self.out.append(nn.ReLU())
            self.out.append(nn.Dropout(dropout_out))

        self.out.append(nn.Linear(nhidden_out, self.nout))
        self.out = nn.Sequential(*self.out)

    def forward(self, x):
        x = self.flipper(x)
        x = self.conv(x)
        return self.out(x)


class CosmoData(Dataset):
    def __init__(self):
        super().__init__()
        #dat_dir = '/home/pl7508/scratch/simbig/mesh'
        dat_dir = '/Users/pablo/Dropbox/data/Simbig/OLD'
        y_train = np.loadtxt(os.path.join(dat_dir, 'simbig.cmass_sgc.theta.dat'), skiprows=1)
        y_train = y_train[:, :5]
        self.x_train = np.load(
                os.path.join(dat_dir, f'simbig.cmass_sgc.mesh.nside{Nside}.npy'),
                mmap_mode='r+'
                )

        #self.x_avg = np.load(os.path.join(dat_dir, f'train_inp_avg.nside{Nside}.npy'), mmap_mode='r+')[0]
        #self.x_std = np.load(os.path.join(dat_dir, f'train_inp_std.nside{Nside}.npy'), mmap_mode='r+')[0]
        self.x_avg = np.mean(self.x_train, axis=0, keepdims=False)
        self.x_std = np.std(self.x_train, axis=0, keepdims=False)
        y_train_new = np.copy(y_train)
        #y_train_new[:,1] = y_train[:,0]/y_train[:,1]
        #y_train_new[:,2] = y_train[:,0]*y_train[:,2]
        #y_train_new[:,4] = y_train[:,4]*y_train[:,0]**0.5
        y_max = np.max(y_train_new, axis=0, keepdims=True)
        y_min = np.min(y_train_new, axis=0, keepdims=True)
        self.y_train = (y_train_new - y_min)/(y_max - y_min)
        self.y_min = y_min
        self.y_max = y_max

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.x_train[idx]
        y = self.y_train[idx, :5]
        x = (x - self.x_avg)/(self.x_std + 1e-20)
        return x, y


def train():

    dataset = CosmoData()

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.seed(seed)
    np.random.shuffle(indices)
    split = 20000
    train_indices, val_indices = indices[:split], indices[split:]
    print('Training and validation indices:', len(train_indices), len(val_indices))

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    batch_size=128

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'data/saved_models/')

    #print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNNModel(n_pars=5, kernel=3, n_comps=1, cov_type="full", Nside=Nside, device=device).cuda()
    #model.load(f"mesh_1comp_nside{Nside}.pt", path=model_path)
    #model.load(f"mesh_1comp_nside{Nside}_pretrained.pt", path=model_path)

    # Pre-training
    model.train(train_loader=train_loader, valid_loader=valid_loader, lr=1e-4, 
                num_epochs=250, num_workers=4,
                pretrain=True, patience=10, save_every=50, 
                save_name="mesh_1comp_nside{Nside}_pretrained.pt", save_path=model_path)

    model.save(f"mesh_1comp_nside{Nside}_pretrained.pt", path=model_path)

    # Make prediction
    data_path = os.path.join(dir_path, 'data/predictions/')

    # SWAG training
    model.train(train_loader=train_loader, valid_loader=valid_loader, lr=5e-4, 
            num_epochs=50, num_workers=4, pretrain=False)

    model.save(f"mesh_1comp_nside{Nside}.pt", path=model_path)

    dat_dir = '/home/pl7508/scratch/simbig/mesh'
    with torch.no_grad():
        for subdir in ['test', 'test_fof', 'test_abacus']:
        #y_test = np.loadtxt(os.path.join(dat_dir, f'simbig.cmass_sgc.{subdir}.theta.dat'), skiprows=1)
            x_test = np.load(os.path.join(dat_dir, f'simbig.cmass_sgc.{subdir}.mesh.nside{Nside}.npy'), mmap_mode='r+')

        # Normalize outputs
        #y_test_normed = (y_test[:,:5] - dataset.y_min[:,:5])/(dataset.y_max[:,:5] - dataset.y_min[:,:5])

            samples = np.zeros([len(x_test), 100, 21])
            for i in range(len(x_test)//50):
            # Normalize inputs
                wx_test = (x_test[50*i:50*(i+1)] - dataset.x_avg[np.newaxis]) / (dataset.x_std[np.newaxis] + 1e-20)
                wx_test = torch.as_tensor(wx_test, dtype=torch.float32, device=device)

                s = model.generate_samples(wx_test, nsamples=100, scale=.5, verbose=True)
                samples[50*i:50*(i+1)] = s.detach().numpy()[:,0]

            np.save(os.path.join(data_path, f"swag_samples_nside{Nside}_{subdir}"), samples)
            torch.cuda.empty_cache()
            gc.collect()


    x_true = np.load(os.path.join(dat_dir, f'mesh.nside{Nside}.hod.z0p5.cmass_sgc.z0p5.observed.npy'))

    x_true = x_true[np.newaxis, np.newaxis, :, :, :]

    wx_true = (x_true - dataset.x_avg[np.newaxis]) / (dataset.x_std[np.newaxis] + 1e-20)
    wx_true = torch.as_tensor(wx_true, dtype=torch.float32, device=device)


    s = model.generate_samples(wx_true, nsamples=100, scale=.5, verbose=True)
    s = s.detach().numpy()
    np.save(os.path.join(data_path, f"swag_samples_nside{Nside}_truth.pt"), s)



if __name__ == "__main__":
    train()

