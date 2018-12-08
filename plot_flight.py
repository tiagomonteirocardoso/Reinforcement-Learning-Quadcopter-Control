'''
plot the flight of the quadcopter like a telemetry.
It includes position (meter), attitude (Euler angles in radians), velocity (m/s), angular velicity (radians/s), rotor speeds (rotations per second) and reward along the simulation time.

code implemented by Diogo Dutra (retrieved from https://github.com/diogodutra/quad_AI)

'''

import matplotlib.pyplot as plt

def plot_flight(results):
    plt.figure(figsize=(20,6))    
    plt.subplot(2, 3, 1)
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(2, 3, 2)
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(2, 3, 3)
    plt.plot(results['time'], results['rotor_speed1'], label='rotor_speed1')
    plt.plot(results['time'], results['rotor_speed2'], label='rotor_speed2')
    plt.plot(results['time'], results['rotor_speed3'], label='rotor_speed3')
    plt.plot(results['time'], results['rotor_speed4'], label='rotor_speed4')
    plt.legend()
    _ = plt.ylim()

    plt.subplot(2, 3, 4)
    plt.plot(results['time'], results['x_velocity'], label='x_velocity')
    plt.plot(results['time'], results['y_velocity'], label='y_velocity')
    plt.plot(results['time'], results['z_velocity'], label='z_velocity')
    plt.legend()
    _ = plt.ylim()


    plt.subplot(2, 3, 5)
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.legend()
    _ = plt.ylim()


    plt.subplot(2, 3, 6)
    plt.plot(results['time'], results['reward'], label='reward per step')
    plt.legend()
    _ = plt.ylim()