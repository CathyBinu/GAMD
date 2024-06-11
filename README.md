# GAMD
This work showcases the application of Graph neural networks (GNN) to simulation particle systems. A neural network architecture called GNN Accelerated Molecular Dynamics(GAMD) is implemented here. 
This model is trained and tested on the data collected from Molecular Dynamics Simulation (MDS) of the Lennnard Jones (LJ) system using the OpenMM library. 


In the datagenerate.py code, we have produced a simulation of the Lennard Jones system using the OpenMM library. Here we have run the simulation under a barostat and thermostat for a number of steps, followed by the removal of the barostat thus forming a thermostat phase, generating an NVT ensemble.  After this the thermostat is removed and the system is allowed to evolve in an NVE phase. The data of the positions of the particles and the forces between them is collected beyond this and saved. 

The datavisualize.py code visualises the data colleced by creating a picture of the particles in the box. A histogram of the velocities is also plotted which is observed to follow the Maxwell velocity distribution. 

The datapreparation.py code, prepares the data for training and testing of the neural network. 

graphneuralnet.py code contains all the functions required in the neural network architecture of GAMD. In encompasses the code for creating 'Bare' features, creating neighbor list, constructing the graph, the GNN layer and the GNN decoder. 


The graphnntrain.py code trains the graphneuralnet using teh data generated above and the loss function vs epochs for training and testing is plotted. 

Finally, in graphnntrain.py, the trained graph neural network is used to predict the forces acting between particles in the test data and the preducted forces are compared with the ground truth forces from the MDS. This is shown via a plot of Predicted Forces vs Ground Truth Forces. 
