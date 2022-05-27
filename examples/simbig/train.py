from cosmoswag import SWAGModel
import torch
from torch import nn
import torchvision
import os

class SWAG_simbig(SWAGModel):
    def __init__(self, npars, ncomps=1, cov_type="diag", kernel=3):
        SWAGModel.__init__(self, 0, npars, ncomps, cov_type)

        self.flipper = nn.Sequential(
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomVerticalFlip(p=0.5)
        )

        self.conv = nn.Sequential(
            nn.Conv3d(1, 8, kernel, padding = 'same'),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool3d(4),
            nn.Conv3d(16, 16, kernel, padding = 'same'),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool3d(4),
            nn.Conv3d(32, 32, kernel, padding = 'same'),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel, padding = 'same'),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )

        self.batchnorm = nn.InstanceNorm3d(64)

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.nout),
        )

    def forward(self, x):
        x = self.flipper(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        return self.out(x)


def train():
    path = '/home/pl7508/scratch/simbig/mesh/'
    x_train = torch.load(os.path.join(path, "train_inp.pt"))
    y_train = torch.load(os.path.join(path, "train_out.pt"))
    x_val = torch.load(os.path.join(path, "val_inp.pt"))
    y_val = torch.load(os.path.join(path, "val_out.pt"))
    low, _ = torch.min(y_train, dim=0, keepdim = True)
    high, _ = torch.max(y_train, dim=0, keepdim = True)
    print('Normalizing')
    print('Low =', low)
    print('High =', high)
    y_train = (y_train - low) / (high - low)
    y_val = (y_val - low) / (high - low)


    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'data/saved_models/')

    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    model = SWAG_simbig(npars=5, kernel=3, ncomps=3, cov_type="full").cuda()
    # model.load("mesh.pt", path=model_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    model = torch.nn.DataParallel(model)
    

    # Pre-training
    model.train(x_train, y_train, lr=1e-4, num_epochs=250, num_workers=4,
                pretrain=True, patience=20, batch_size=16, save_every=50, 
                save_name="mesh_pretrained.pt", save_path=model_path)

    model.save("mesh_pretrained.pt", path=model_path)
    
    # SWAG training
    model.train(x_train, y_train, lr=2e-4, num_epochs=50, num_workers=4,
                pretrain=False, batch_size=16)

    #y_pred = model.forward(x_val.cuda())
    #y_pred = y_pred*(high[0]-low[0]) + low[0]
    #torch.save(y_pred, os.path.join(path, "val_pred.pt"))

if __name__ == "__main__":
    train()

