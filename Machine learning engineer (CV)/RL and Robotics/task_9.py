from sim_class import Simulation
import random
import os

# set current directory to working directory
os.chdir(os.path.dirname(os.path.realpath(__file__)))

sim = Simulation(num_agents=1, render=False)

def run_simulation(sim, iterations, velocity_params):

    for _ in range(iterations):
        velocity_x, velocity_y, velocity_z, drop_command = velocity_params

        actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

        state = sim.run(actions)

        coordinates = state['robotId_1']['pipette_position']

    return coordinates

movement_directions = [
    ("Move Bottom Right Down", (0.5, 0.5, -0.5, 0), 500),
    ("Move Bottom Right Up", (0, 0, 0.5, 0), 100),
    ("Move Top Left Up", (-0.5, -0.5, 0, 0), 500),
    ("Move Top Left Down", (0, 0, -0.5, 0), 100),
    ("Move Bottom Left Down", (0.5, 0, 0, 0), 500),
    ("Move Bottom Left Up", (0, 0, 0.5, 0), 100),
    ("Move Top Right Up", (-0.5, 0.5, 0, 0), 500),
    ("Move Top Right Down", (0, 0, -0.5, 0), 100),
]

X = []
Y = []
Z = []

for movement_name, velocity_params, iterations in movement_directions:
    pipette_position = run_simulation(sim, iterations, velocity_params)
    print(f'After {movement_name}:')
    print(f'X Coordinate: {pipette_position[0]}')
    print(f'Y Coordinate: {pipette_position[1]}')
    print(f'Z Coordinate: {pipette_position[2]}')
    X.append(pipette_position[0])
    Y.append(pipette_position[1])
    Z.append(pipette_position[2])

print(f'X Coordinate: {max(X)} (Max), {min(X)} (Min)')
print(f'Y Coordinate: {max(Y)} (Max), {min(Y)} (Min)')
print(f'Z Coordinate: {max(Z)} (Max), {min(Z)} (Min)')

sim.reset(num_agents=1)