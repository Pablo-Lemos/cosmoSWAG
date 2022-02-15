#from cmb_simulator import read_data
#from swag import SWAGModel
from read_simulations import read_data
from swag_galaxies import SWAGModelGal
import torch

def train():
    x_train, y_train, x_val, y_val, delta_y = read_data()
    low, _ = torch.min(y_train, dim=0, keepdim = True)
    high, _ = torch.max(y_train, dim=0, keepdim = True)
    print('Normalizing')
    print('Low =', low)
    print('High =', high)
    y_train = (y_train - low) / (high - low)
    y_val = (y_val - low) / (high - low)

    nin = x_train.shape[1]
    npars = y_train.shape[1]
    model = SWAGModelGal()

    # Pre-training
    model.train(x_train, y_train, lr=1e-4, num_epochs=50000, num_workers=0,
                pretrain=True, patience=100)

    # Swag training
    model.train(x_train, y_train, lr=2e-4, num_epochs=1000, num_workers=0,
                pretrain=False)

    model.save("galaxies_v3.pt")


if __name__ == "__main__":
    train()

