# Task 9 

| Coordinate | Maximum Value | Minimum Value |
|------------|---------------|---------------|
| X          | 0.253         | -0.187        |
| Y          | 0.2196        | -0.1705       |
| Z          | 0.2896        | 0.1201        |

The provided Python script simulates a robotic environment using a custom Simulation class. The script begins by importing necessary modules, including Simulation from the sim_class module, as well as random and os.

The working directory is then set to the directory containing the script to ensure that relative paths are resolved correctly. The simulation is initialized with an instance of the Simulation class, configured for a single agent and with rendering disabled.

A function, run_simulation, is defined to execute the simulation for a specified number of iterations with given velocity parameters, returning the final pipette position.

The script defines a list of movement directions, each consisting of a movement name, velocity parameters, and the number of iterations. It then iterates through these directions, running the simulation for each and printing the resulting pipette position. The X, Y, and Z coordinates of the pipette positions are collected in separate lists.

After completing all simulations, the script prints statistics on the maximum and minimum X, Y, and Z coordinates obtained during the simulation. Finally, the simulation is reset with one agent.

It's important to note that the details of the Simulation class and its dependencies are not provided in the code snippet, making it necessary to inspect the implementation of that class to fully understand the simulation environment and dependencies involved. The simulation seems to represent a robotic system with a pipette, and the script is designed to analyze pipette positions resulting from movements in different directions.

