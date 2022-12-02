from data_object import read_binned_data
from cosmoswag import SWAGModel
import torch
import os 
import numpy as np

class SWAG_CMB(SWAGModel):
    def __init__(self, nin, npars, ncomps=1, cov_type="diag", nHidden = 128, nLayers=2):
        #super(self.__class__, self).__init__(nin, npars, ncomps, cov_type)
        SWAGModel.__init__(self, nin, npars, ncomps, cov_type)

        layers = (
            [torch.nn.Linear(nin, nHidden, device=self._device), torch.nn.ReLU()]
            + [[torch.nn.Linear(nHidden, nHidden, device=self._device), torch.nn.ReLU()][i%2] for i
               in range(nLayers)]
            + [torch.nn.Linear(nHidden, self.nout, device=self._device)]
        )
        self.out = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


def train():
    data = read_binned_data(path="data/cmb_sims_10k/")
    data.read_truth(filename="planck_binned.txt")
    x_train, y_train, x_val, y_val = data.get_data()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir_path, 'data/cmb_sims_10k/cls_binned_cov.npy')
    model_path = os.path.join(dir_path, 'data/saved_models/')
    cov_x = np.load(path)
    cov_x = torch.tensor(cov_x, dtype = torch.float32)
    cov_x = data.normalize_covariance(cov_x) 

    nin = x_train.shape[1]
    npars = y_train.shape[1]
    model = SWAG_CMB(nin=nin, npars=npars, ncomps=1, cov_type="full", nHidden=128, nLayers=6)

    # Pre-training
    model.train(x_train, y_train, cov_x=cov_x, lr=1e-3, num_epochs=10000,
                num_workers=0, pretrain=True, patience=20, batch_size=100)

    model.save("cmb_gmn_binned_pretrained_10ksims.pt", path=model_path)

    # Swag training
    model.train(x_train, y_train, cov_x=cov_x, lr=1e-4, num_epochs=100,
                num_workers=0,
                pretrain=False)

    model.save("cmb_gmn_binned_10ksims.pt", path=model_path)


if __name__ == "__main__":
    train()

