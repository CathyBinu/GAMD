from openmmtools import testsystems
from simtk import openmm, unit
from openmm import app
from openmmtools import integrators
import os
import numpy as np
from openmmtools.constants import kB
import matplotlib.pyplot as plt
from tqdm import tqdm

n_particles = 258
m0 = 39.9 * unit.dalton
e0 = 0.238 * unit.kilocalorie_per_mole
l0 = 3.4 * unit.angstrom
v0 = np.sqrt(e0/m0)
t0 = l0 / v0
T0 = e0 / kB
f0 = m0 * l0 / t0**2
pressure = 1.0 * unit.atmospheres
temperature = 80. * unit.kelvin
delta_t = t0 * 1e-3
cutoff = 3. * l0

print(f'Natural length scale: {l0}')
print(f'Natual time scale: {t0}')
print(f'Natural velocity scale: {v0}')
print(f'Natural temperature scale: {T0}')


natural_velocity_scale_value = v0.value_in_unit(unit.meter / unit.second)
print(natural_velocity_scale_value)
natural_length_scale_value = l0.value_in_unit(unit.nanometer)
print(natural_length_scale_value)
natural_force_scale_value = f0.value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
print(natural_force_scale_value)


box_size = 6. * (n_particles/4000)**(1/3.) * unit.nanometer

chain_length = 10
friction = 25. / unit.picosecond
num_mts = 5
num_yoshidasuzuki = 5


fluid = testsystems.LennardJonesFluid(nparticles=n_particles, reduced_density=0.5, cutoff=cutoff, shift=True)
fluid.system.setDefaultPeriodicBoxVectors(
    (box_size, 0, 0),
    (0, box_size, 0),
    (0, 0, box_size)
)
for force in fluid.system.getForces():
    if isinstance(force, openmm.NonbondedForce):
        force.setNonbondedMethod(openmm.NonbondedForce.PME)

barostat = openmm.MonteCarloBarostat(pressure, temperature, 25)
fluid.system.addForce(barostat)
[topology, system, positions] = fluid.topology, fluid.system, fluid.positions
integrator = integrators.NoseHooverChainVelocityVerletIntegrator(system, temperature, friction, delta_t, chain_length, num_mts)
simulation = app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)
simulation.minimizeEnergy()

nsteps_baro = 20000

print("Running barostat...")
for _ in tqdm(range(nsteps_baro)):
    simulation.step(1)

fluid.system.removeForce(fluid.system.getNumForces() - 1)


nsteps_thermo = 20000

print("\nRunning thermostat...")
for t in tqdm(range(nsteps_thermo)):
    simulation.step(1)

# Retrieve state after thermostat phase
final_state_thermo = simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
positions_thermo = final_state_thermo.getPositions()
velocities_thermo = final_state_thermo.getVelocities()
box_vectors_thermo = final_state_thermo.getPeriodicBoxVectors()

# Create a new simulation for the NVE phase
nve_integrator = openmm.VerletIntegrator(delta_t)
nve_simulation = app.Simulation(topology, system, nve_integrator)
nve_simulation.context.setPositions(positions_thermo)
nve_simulation.context.setVelocities(velocities_thermo)
nve_simulation.context.setPeriodicBoxVectors(*box_vectors_thermo)


# Run NVE phase
nsteps_nve = 600000
data_interval = 50
velocities_array = np.zeros((nsteps_nve // data_interval, n_particles, 3))
positions_array = np.zeros((nsteps_nve // data_interval, n_particles, 3))
forces_array = np.zeros((nsteps_nve // data_interval, n_particles, 3))
data_index = 0

print("\nRunning NVE simulation...")

for t in tqdm(range(nsteps_nve)):
    if t % data_interval == 0:
        state = nve_simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, enforcePeriodicBox=True)
        velocities_array[data_index] = state.getVelocities(asNumpy=True).value_in_unit(unit.meter / unit.second)
        positions_array[data_index] = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        forces_array[data_index] = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole/unit.nanometer)
        data_index += 1
    nve_simulation.step(1)

# Retrieve final box size after NVE phase

box_vectors_final = state.getPeriodicBoxVectors()
box_size_final = box_vectors_final[0][0].value_in_unit(unit.nanometer)  # Assuming a cubic box
print(f"Final box size: {box_size_final} nm")

volume = state.getPeriodicBoxVolume()
mass = fluid.system.getParticleMass(0) * n_particles  # Assuming all particles have the same mass
density = mass / volume

print(density)


box_size_dimensionless = box_size_final / natural_length_scale_value
cutoff_dimensionless = cutoff / l0
delta_t_dimensionless = delta_t/t0
temperature_dimensionless = temperature / T0
velocities_dimensionless = velocities_array / natural_velocity_scale_value
positions_dimensionless = positions_array / natural_length_scale_value
forces_dimensionless = forces_array / natural_force_scale_value
print(box_size_dimensionless)
print(cutoff_dimensionless)
print(delta_t_dimensionless)
print(temperature_dimensionless)


os.makedirs(f'./lj_thermo_baro_data', exist_ok=True)
np.savez(f'./lj_thermo_baro_data/train_test_positions_velocities_forces.npz', positions=positions_dimensionless[0:10000], velocities=velocities_dimensionless[0:10000], 
         forces=forces_dimensionless[0:10000])
np.savez(f'./lj_thermo_baro_data/validation_positions_velocities_forces.npz', positions=positions_dimensionless[10000:], velocities=velocities_dimensionless[10000:], 
         forces=forces_dimensionless[10000:])

global_data = {"cutoff" : {"value": cutoff_dimensionless, "description": "Cutoff in natural units"}, 
               "box_size": {"value": box_size_dimensionless, "description": "Equilibrium box size in natural units"}, 
               "delta_t": {"value": delta_t_dimensionless, "description": "delta_t in natural units"}, 
               "T": {"value": temperature_dimensionless, "description": "Dimensionless temperature"}}

with open("./lj_thermo_baro_data/global_data.txt", "w") as file:
    for var, details in global_data.items():
        file.write(f"{var}: {details['value']} ({details['description']})\n")