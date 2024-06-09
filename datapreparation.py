import numpy as np
import os
from sklearn.model_selection import train_test_split


def train_test_split_shuffled(positions, forces, test_size=0.2, random_state=None):
    # Ensure that positions and forces have the same first dimension
    assert positions.shape[0] == forces.shape[0], "The first dimension of positions and forces must be the same."

    # Get the number of instances (T)
    T = positions.shape[0]

    # Create an array of indices and shuffle it
    indices = np.arange(T)
    np.random.seed(random_state)
    np.random.shuffle(indices)

    # Shuffle positions and forces using the shuffled indices
    positions_shuffled = positions[indices]
    forces_shuffled = forces[indices]

    # Perform the train-test split
    positions_train, positions_test, forces_train, forces_test = train_test_split(
        positions_shuffled, forces_shuffled, test_size=test_size, random_state=random_state
    )

    return positions_train, positions_test, forces_train, forces_test




data = np.load('./lj_thermo_baro_data/train_test_positions_velocities_forces.npz')
positions = data['positions']
velocities = data['velocities']
forces = data['forces']


positions_train, positions_test, forces_train, forces_test = train_test_split_shuffled(positions, forces)


os.makedirs(f'./train_test_data', exist_ok=True)
np.savez(f'./train_test_data/positions_forces.npz', train_pos=positions_train, test_pos = positions_test, 
         train_forces=forces_train, test_forces=forces_test)