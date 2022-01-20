from cmb_simulator import read_data
from swag import SWAGModel

def train():
    x_train, y_train, x_val, y_val, delta_y = read_data()
    nin = x_train.shape[1]
    npars = y_train.shape[1]
    model = SWAGModel(nin=nin, npars=npars)

    # Pre-training
    model.train(x_train, y_train, lr=1e-4, num_epochs=5000, num_workers=8,
                pretrain=True, patience=20)

    # Swag training
    model.train(x_train, y_train, lr=1e-4, num_epochs=100, num_workers=8,
                pretrain=False)

    model.save("cmb_noiseless.pt")


if __name__ == "__main__":
    train()

