import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class DCMotorPlant:
    def __init__(self, R=1.0, L=0.5, Kb=0.01, Kt=0.01, J=0.01, b=0.1):
        self.R = R      # Armature resistance (ohms)
        self.L = L      # Armature inductance (H)
        self.Kb = Kb    # Back EMF constant (V/rad/s)
        self.Kt = Kt    # Motor torque constant (Nm/A)
        self.J = J      # Rotor inertia (kg.m^2)
        self.b = b      # Damping coefficient (Nms)
        self.noise_params = {'V': 0.0, 'i': 0.0, 'omega': 0.0}  # Default no noise

    def set_noise(self, V_noise=0.0, i_noise=0.0, omega_noise=0.0):
        """Set noise levels for voltage, current, and angular velocity."""
        self.noise_params = {'V': V_noise, 'i': i_noise, 'omega': omega_noise}

    def motor_model(self, y, t, V):
        i, omega = y
        # Add noise if specified
        V += np.random.normal(0, self.noise_params['V']) if self.noise_params['V'] > 0 else 0.0  # Voltage noise in volts (V)
        i += np.random.normal(0, self.noise_params['i']) if self.noise_params['i'] > 0 else 0.0  # Current noise in amperes (A)
        omega += np.random.normal(0, self.noise_params['omega']) if self.noise_params['omega'] > 0 else 0.0  # Angular velocity noise in radians per second (rad/s)

        di_dt = (V - self.R * i - self.Kb * omega) / self.L
        domega_dt = (self.Kt * i - self.b * omega) / self.J
        return [di_dt, domega_dt]

    def simulate(self, t, initial_conditions, V_func):
        solution = odeint(self.motor_model, initial_conditions, t, args=(V_func,))
        i, omega = solution.T
        return i, omega

    def plot_results(self, t, omega, omega_desired, i, V):
        # Integrate omega to get the position
        position_actual = np.cumsum(omega) * (t[1] - t[0])
        
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(t, position_actual, label='Actual Position (rad)')
        plt.plot(t, np.sin(t), label='Desired Position (rad)', linestyle='dashed')
        plt.xlabel('Time (s)')
        plt.ylabel('Position (rad)')
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(t, omega, label='Actual Angular Velocity (rad/s)')
        plt.plot(t, omega_desired, label='Desired Angular Velocity (rad/s)', linestyle='dashed')
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
