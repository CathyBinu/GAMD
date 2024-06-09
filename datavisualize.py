import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



data = np.load('./lj_thermo_baro_data/train_test_positions_velocities_forces.npz')
positions = data['positions']
velocities = data['velocities']


box_size = 6.717201743984198

final_positions = positions[-1, :, :]

# Convert positions to a NumPy array for easier manipulation
pos_array = np.array([[pos[0], pos[1], pos[2]] for pos in final_positions])

# Extract x, y, and z coordinates of wracpped positions
x_ = pos_array[:, 0]
y_ = pos_array[:, 1]
z_ = pos_array[:, 2]

# Create a 3D scatter plot using wrapped positions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_, y_, z_)

# Define the vertices of the cube (box boundaries)
vertices = [
    [0, 0, 0],
    [0, box_size, 0],
    [box_size, box_size, 0],
    [box_size, 0, 0],
    [0, 0, box_size],
    [0, box_size, box_size],
    [box_size, box_size, box_size],
    [box_size, 0, box_size]
]

# Define the edges of the cube (box boundaries)
edges = [
    [vertices[0], vertices[1], vertices[2], vertices[3], vertices[0]],
    [vertices[4], vertices[5], vertices[6], vertices[7], vertices[4]],
    [vertices[0], vertices[4]],
    [vertices[1], vertices[5]],
    [vertices[2], vertices[6]],
    [vertices[3], vertices[7]]
]

# Plot the edges of the cube
for edge in edges:
    ax.plot3D(*zip(*edge), color="black")

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([0, box_size])
ax.set_ylim([0, box_size])
ax.set_zlim([0, box_size])

# Show the plot
plt.show()



speed_array = np.linalg.norm(velocities, axis=2)
speeds = speed_array.reshape(-1)
print(speeds.shape)


def Maxwell(v, T):
    return 4.*np.pi * ((1./(2*np.pi*T))**(1.5)) * (v**2) * np.exp(-v**2 /(2.*T))

test_speds = np.arange(0, 6, 0.01)
T = 0.6679678180305316
actual_dist = Maxwell(test_speds, T)
plt.plot(test_speds, actual_dist)
plt.hist(speeds, facecolor='g', edgecolor="white", density=True, alpha=0.5, bins=100)
plt.xlabel('u')
plt.ylabel('P(u)');




