from data_object import read_data
from cosmoswag import SWAGModel
import torch

class SWAG_CMB(SWAGModel):
    def __init__(self, nin, npars, ncomps=1, cov_type="diag", nHidden = 128):
        #super(self.__class__, self).__init__(nin, npars, ncomps, cov_type)
        SWAGModel.__init__(self, nin, npars, ncomps, cov_type)

        layers = (
            [torch.nn.Linear(nin, nHidden, device=self._device), torch.nn.ReLU()]
            + [[torch.nn.Linear(nHidden, nHidden, device=self._device), torch.nn.ReLU()][i%2] for i
               in range(2*2*2)]
            + [torch.nn.Linear(nHidden, self.nout, device=self._device)]
        )
        self.out = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.out(x)


def train():
    data = read_data()
    data.read_truth(filename="simulated_truth.txt")
    x_train, y_train, x_val, y_val = data.get_data()
    delta_x = data.get_error()
    nin = x_train.shape[1]
    npars = y_train.shape[1]
    model = SWAG_CMB(nin=nin, npars=npars, ncomps=3,cov_type="full")

    # Pre-training
    model.train(x_train, y_train, delta_x=delta_x, lr=1e-4, num_epochs=1000, \
                                                                num_workers=0,
                 pretrain=True, patience=20)

    model.save("cmb_gmn_v3_pretrained.pt")

    # Swag training
    model.train(x_train, y_train, delta_x=delta_x, lr=1e-4, num_epochs=100,
                num_workers=0,
                pretrain=False)

    model.save("cmb_gmn_v3.pt")


if __name__ == "__main__":
    train()

