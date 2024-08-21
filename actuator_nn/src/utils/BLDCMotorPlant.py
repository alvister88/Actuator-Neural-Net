import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class BLDCMotorPlant:
    def __init__(self, Rs=1.0, Ld=0.5, Lq=0.5, Kb=0.01, Kt=0.2, J=0.01, b=0.1, poles=42):
        self.Rs = Rs      # Stator resistance (ohms)
        self.Ld = Ld      # Direct-axis inductance (H)
        self.Lq = Lq      # Quadrature-axis inductance (H)
        self.Kb = Kb      # Back EMF constant (V/rad/s)
        self.Kt = Kt      # Torque constant (Nm/A)
        self.J = J        # Rotor inertia (kg.m^2)
        self.b = b        # Damping coefficient (Nms)
        self.poles = poles  # Number of poles
        self.noise_params = {'Vd': 0.0, 'Vq': 0.0, 'omega': 0.0}  # Default no noise

    def set_noise(self, Vd_noise=0.0, Vq_noise=0.0, omega_noise=0.0):
        """Set noise levels for d-axis voltage, q-axis voltage, and angular velocity."""
        self.noise_params = {'Vd': Vd_noise, 'Vq': Vq_noise, 'omega': omega_noise}

    def motor_model(self, y, t, Vd_func, Vq_func):
        id, iq, omega = y
        Vd = Vd_func(t)
        Vq = Vq_func(t)

        # Add noise if specified
        Vd += np.random.normal(0, self.noise_params['Vd']) if self.noise_params['Vd'] > 0 else 0.0  # d-axis voltage noise
        Vq += np.random.normal(0, self.noise_params['Vq']) if self.noise_params['Vq'] > 0 else 0.0  # q-axis voltage noise
        omega += np.random.normal(0, self.noise_params['omega']) if self.noise_params['omega'] > 0 else 0.0  # Angular velocity noise

        # Electrical dynamics
        did_dt = (Vd - self.Rs * id + self.Lq * omega * iq) / self.Ld
        diq_dt = (Vq - self.Rs * iq - self.Ld * omega * id - self.Kb * omega) / self.Lq
        
        # Mechanical dynamics
        torque = self.Kt * iq
        domega_dt = (torque - self.b * omega) / self.J

        return [did_dt, diq_dt, domega_dt]

    def simulate(self, t, initial_conditions, Vd_func, Vq_func):
        solution = odeint(self.motor_model, initial_conditions, t, args=(Vd_func, Vq_func))
        id, iq, omega = solution.T
        return id, iq, omega

    def plot_results(self, t, omega, omega_desired, id, iq, Vd, Vq):
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
        plt.plot(t, id, label='d-axis Current (A)')
        plt.plot(t, iq, label='q-axis Current (A)')
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.grid()
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(t, Vd, label='d-axis Voltage (V)', color='r')
        plt.plot(t, Vq, label='q-axis Voltage (V)', color='b')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    bldc_motor = BLDCMotorPlant(Rs=0.5, Ld=0.01, Lq=0.01, Kb=0.01, Kt=0.01, J=0.01, b=0.001)
    bldc_motor.set_noise(Vd_noise=0.001, Vq_noise=0.001, omega_noise=0.0001)
    
    t = np.linspace(0, 10, 1000)
    initial_conditions = [0.0, 0.0, 0.0]  # [id, iq, omega]
    
    Vd_func = lambda t: 1.0  # Constant d-axis voltage
    Vq_func = lambda t: 1.0  # Constant q-axis voltage
    
    id, iq, omega = bldc_motor.simulate(t, initial_conditions, Vd_func, Vq_func)
    
    omega_desired = np.gradient(np.sin(t), t[1] - t[0])
    
    # Evaluate Vd and Vq over the entire time vector
    Vd_values = np.array([Vd_func(time) for time in t])
    Vq_values = np.array([Vq_func(time) for time in t])
    
    bldc_motor.plot_results(t, omega, omega_desired, id, iq, Vd_values, Vq_values)

