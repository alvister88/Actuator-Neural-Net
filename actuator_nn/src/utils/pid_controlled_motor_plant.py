from DCMotorPlant import DCMotorPlant
from PIDController import PIDController
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def main():
    motor = DCMotorPlant(R=1.0, L=0.5, Kb=0.01, Kt=0.01, J=0.01, b=0.1)
    
    # Set noise levels (example: V_noise = 0.1, i_noise = 0.01, omega_noise = 0.05)
    motor.set_noise(V_noise=0.001, i_noise=0.0001, omega_noise=0.0001)
    
    # Time vector
    t = np.linspace(0, 10, 1000)
    dt = t[1] - t[0]
    
    # Desired position trajectory (sine wave)
    position_desired = np.sin(t)
    # Desired velocity trajectory (derivative of position)
    omega_desired = np.gradient(position_desired, dt)

    # PID Controller
    pid = PIDController(Kp=500.0, Ki=1.0, Kd=5.0, setpoint=0)

    # Simulate the motor
    i = np.zeros_like(t)
    omega = np.zeros_like(t)
    V = np.zeros_like(t)
    y = [0.0, 0.0]  # Initial conditions: [current, angular velocity]

    for idx in range(1, len(t)):
        pid.setpoint = omega_desired[idx]
        dt = t[idx] - t[idx-1]
        V[idx] = pid.compute(omega[idx-1], dt)
        y = odeint(motor.motor_model, y, [t[idx-1], t[idx]], args=(V[idx],))[1]
        i[idx], omega[idx] = y

    # Plot the results
    plot_results(t, position_desired, omega, omega_desired, i, V)

def plot_results(t, position_desired, omega, omega_desired, i, V):
    # Integrate omega to get the position
    position_actual = np.cumsum(omega) * (t[1] - t[0])
    
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, position_desired, label='Desired Position (rad)')
    plt.plot(t, position_actual, label='Actual Position (rad)', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (rad)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(t, omega_desired, label='Desired Angular Velocity (rad/s)')
    plt.plot(t, omega, label='Actual Angular Velocity (rad/s)', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, i, label='Current (A)')
    plt.xlabel('Time (s)')
    plt.ylabel('Current (A)')
    plt.grid()
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, V, label='Input Voltage (V)', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()
