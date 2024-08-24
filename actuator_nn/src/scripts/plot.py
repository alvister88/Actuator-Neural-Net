import pandas as pd
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path, delimiter=',')

def plot_data(data):
    # Create a figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(12, 20), sharex=True)
    fig.suptitle('Actuator Data Visualization', fontsize=16)

    # Create a sample index
    samples = range(len(data))

    # Plot Samples vs TargetPosition
    axs[0].plot(samples, data['TargetPosition'], label='Target Position')
    axs[0].plot(samples, data['CurrentPosition'], label='Current Position')
    axs[0].set_ylabel('Position')
    axs[0].legend()
    axs[0].grid(True)

    # Plot Samples vs Velocity
    axs[1].plot(samples, data['Velocity'], label='Velocity')
    axs[1].set_ylabel('Velocity')
    axs[1].legend()
    axs[1].grid(True)

    # Plot Samples vs Acceleration
    axs[2].plot(samples, data['Acceleration'], label='Acceleration')
    axs[2].set_ylabel('Acceleration')
    axs[2].legend()
    axs[2].grid(True)

    # Plot Samples vs Error
    axs[3].plot(samples, data['Error'], label='Error')
    axs[3].set_ylabel('Error')
    axs[3].legend()
    axs[3].grid(True)

    # Plot Samples vs Torque
    axs[4].plot(samples, data['Torque'], label='Torque')
    axs[4].set_xlabel('Samples')
    axs[4].set_ylabel('Torque')
    axs[4].legend()
    axs[4].grid(True)

    plt.tight_layout()
    plt.show()

def main():
    # Load the data
    data = load_data('../data/normal1.txt')
    
    # Plot the data
    plot_data(data)

if __name__ == "__main__":
    main()