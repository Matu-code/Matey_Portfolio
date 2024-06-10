from simple_pid import PID
from ot2_env_wrapper import OT2Env
import time

# Setting gains for the controllers
Kp = 15
Ki = 0
Kd = 3


env = OT2Env() # Defining the environment

# Setting 3 controllers for each axis
pid_controller_x = PID(Kp=Kp, Ki=Ki, Kd=Kd)
pid_controller_y = PID(Kp=Kp, Ki=Ki, Kd=Kd)
pid_controller_z = PID(Kp=Kp, Ki=Ki, Kd=Kd)

# Taking observations
observation, _ = env.reset()

x, y, z = 0.10775, 0.062, 0.157

# Setting the goal position
pid_controller_x.setpoint = x
pid_controller_y.setpoint = y
pid_controller_z.setpoint = z

for i in range(5000):
    
    # Gaining the current position
    current_x = observation[0]
    current_y = observation[1]
    current_z = observation[2]

    # Calculating the error for each controller
    control_output_x = pid_controller_x(current_x)
    control_output_y = pid_controller_y(current_y)
    control_output_z = pid_controller_z(current_z)

    # Printing for sanity check
    print(control_output_x, control_output_y, control_output_z)

    # Giving actions to the environment
    observation, reward, terminated, truncated, info = env.step([control_output_x, control_output_y, control_output_z])
