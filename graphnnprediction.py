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
from graphnntrain import MolecularDynamicsDataset


if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)


cutoff = 3.0
box_size = 6.717201743984198


model = MDNet(encoding_size=128, out_feat=3, box_size=box_size, cutoff=cutoff)
model = model.to(device)
model.load_state_dict(torch.load('./Saved_Models/Model_Reduced_Edge_Embedding1.pt'))
model.eval()


data = np.load(f'./lj_thermo_baro_data/validation_positions_velocities_forces.npz')

positions = data['positions']
forces = data['forces']
dataset = MolecularDynamicsDataset(positions, forces)
dataloader = DataLoader(dataset, batch_size=1)


force_pred = []
force_gt = []

with torch.no_grad():
    count = 0 
    for i_batch, batch in enumerate(dataloader):
        pos_sample, force_sample = batch
        pos_sample = pos_sample[0].to(device)
        force_predicted = model.predict_forces(pos_sample).detach().cpu().numpy()
        force_gt += [force_sample[0]]
        force_pred += [force_predicted]
        if count%200 == 0:
            print(f'Finished {i_batch}')
        count += 1


predicted_forces = np.concatenate([force.reshape(-1, 3) for force in force_pred]).reshape(-1)
ground_truth_forces = np.concatenate([force.reshape(-1, 3) for force in force_gt]).reshape(-1)


plt.scatter(predicted_forces, ground_truth_forces, color='b', s=1)
min_force = min(predicted_forces.min(), ground_truth_forces.min())
max_force = max(predicted_forces.max(), ground_truth_forces.max())


plt.plot([min_force, max_force], [min_force, max_force], color='red', linestyle='--', label='y=x')

plt.xlabel('Predicted Forces')
plt.ylabel('Ground Truth Forces')
plt.title('Scatter Plot of Predicted vs Ground Truth Forces')
plt.legend();
