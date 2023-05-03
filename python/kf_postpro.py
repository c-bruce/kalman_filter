# Script for working out Kalman filter implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GET DATA
df = pd.read_csv('KF_001_mpu6050.csv')

# CONSTANTS
dt = 0.004 # [s]
freq = 1/dt # [Hz]

def get_time(df, dt, freq):
    return np.arange(0, len(df)/freq, dt)

def vectorized_integration(input_vector, dt):
    output_vector = [0]
    for i in range(len(input_vector)):
        output_vector.append(output_vector[i] + (input_vector[i] * dt))
    return np.array(output_vector[1:])

time = get_time(df, dt, freq)
z_0 = df['z_0'].to_numpy()
z_1 = df['z_1'].to_numpy()
z_2 = df['z_2'].to_numpy()
z_3 = df['z_3'].to_numpy()
x_0 = df['x_0'].to_numpy()
x_1 = df['x_1'].to_numpy()
x_2 = df['x_2'].to_numpy()
x_3 = df['x_3'].to_numpy()
y_0 = df['y_0'].to_numpy()
y_1 = df['y_1'].to_numpy()
y_2 = df['y_2'].to_numpy()
y_3 = df['y_3'].to_numpy()
epsilon = df['epsilon'].to_numpy()

# Plot
fig1, ax1 = plt.subplots(3)
ax1[0].plot(time, z_0)
ax1[0].plot(time, x_0)
ax1[0].plot(time, y_0)
# ax1[0].plot(time, vectorized_integration(x_2, dt))
ax1[1].plot(time, z_2)
ax1[1].plot(time, x_2)
ax1[1].plot(time, y_2)
ax1[2].plot(time, epsilon)
ax1[0].grid()
ax1[1].grid()
ax1[2].grid()
ax1[0].set_ylabel("Roll [deg]")
ax1[1].set_ylabel("Roll rate [deg/s]")
ax1[2].set_ylabel("epsilon")

fig2, ax2 = plt.subplots(3)
ax2[0].plot(time, z_1)
ax2[0].plot(time, x_1)
ax2[0].plot(time, y_1)
# ax2[0].plot(time, vectorized_integration(x_3, dt))
ax2[1].plot(time, z_3)
ax2[1].plot(time, x_3)
ax2[1].plot(time, y_3)
ax2[2].plot(time, epsilon)
ax2[0].grid()
ax2[1].grid()
ax2[2].grid()
ax2[0].set_ylabel("Pitch [deg]")
ax2[1].set_ylabel("Pitch rate [deg/s]")
ax2[2].set_ylabel("epsilon")

plt.show()