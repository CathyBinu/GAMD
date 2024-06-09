import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl
import dgl.function as fn
from graphneuralnet import MDNet


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)




class MolecularDynamicsDataset(Dataset):
    def __init__(self, positions, forces):
        assert positions.shape == forces.shape, "Positions and forces must have the same shape"
        self.positions = torch.tensor(positions, dtype=torch.float32)
        self.forces = torch.tensor(forces, dtype=torch.float32)

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        position = self.positions[idx]
        force = self.forces[idx]
        return position, force
    


def train_validate(model, train_pos, train_forces, test_pos, test_forces, n_epochs, batch_size, lr, alpha_f, device='cpu'):

    dataset_train = MolecularDynamicsDataset(train_pos, train_forces)
    dataset_test = MolecularDynamicsDataset(test_pos, test_forces)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gamma = (alpha_f / lr) ** (1 / n_epochs / 2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.L1Loss()

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(n_epochs)):
        train_loss_epoch = 0
        model.train()
        for batch in dataloader_train:
            positions, forces = batch
            positions = positions.float().to(device)
            forces = forces.float().to(device)
            pos_lst = list(positions.unbind(dim=0))
            force_lst = list(forces.unbind(dim=0))
            optimizer.zero_grad()
            loss = model.training_step(pos_lst, force_lst, criterion)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss.item()
        train_loss_epoch /= len(dataloader_train)
        train_losses.append(train_loss_epoch)
        if epoch >= n_epochs // 2:
            scheduler.step()

        test_loss_epoch = 0
        model.eval()
        with torch.no_grad():
            for batch in dataloader_test:
                positions, forces = batch
                positions = positions.float().to(device)
                forces = forces.float().to(device)
                pos_lst = list(positions.unbind(dim=0))
                force_lst = list(forces.unbind(dim=0))
                loss = model.training_step(pos_lst, force_lst, criterion)
                test_loss_epoch += loss.item()
        test_loss_epoch /= len(dataloader_test)
        test_losses.append(test_loss_epoch)

    return train_losses, test_losses




data = np.load(f'./train_test_data/positions_forces.npz')

train_pos = data['train_pos']
train_forces = data['train_forces']

test_pos = data['test_pos']
test_forces = data['test_forces']



train_pos_tsr = torch.tensor(train_pos).float().to(device)
train_forces_tsr = torch.tensor(train_forces).float().to(device)

test_pos_tsr = torch.tensor(test_pos).float().to(device)
test_forces_tsr = torch.tensor(test_forces).float().to(device)


cutoff = 3.0
box_size = 6.717201743984198


model = MDNet(encoding_size=128, out_feat=3, box_size=box_size, cutoff=cutoff)
model = model.to(device)

train_losses, test_losses = train_validate(model, train_pos_tsr, train_forces_tsr, test_pos_tsr, test_forces_tsr, 
                                           n_epochs=200, batch_size=10, lr=3e-4, alpha_f=1e-7, device=device)






plt.title("Loss vs Epoch")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.plot(train_losses, label='Traiining')
plt.plot(test_losses, label='Testing')
plt.legend();


torch.save(model.state_dict(), "./Saved_Models/Model_Reduced_Edge_Embedding1.pt")