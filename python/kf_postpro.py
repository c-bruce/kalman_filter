# Script for working out Kalman filter implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GET DATA
df = pd.read_csv('KF_003_mpu6050.csv')

# CONSTANTS
dt = 0.004 # [s]
freq = 1/dt # [Hz]
g = 9.81 # [m/s**2]

def get_time(df, dt, freq):
    return np.arange(0, len(df)/freq, dt)

def vectorized_integration(input_vector, dt):
    output_vector = [0]
    for i in range(len(input_vector)):
        output_vector.append(output_vector[i] + (input_vector[i] * dt))
    return np.array(output_vector[1:])

time = get_time(df, dt, freq)
z_k_0 = df['z_k_0'].to_numpy()
z_k_1 = df['z_k_1'].to_numpy()
z_k_2 = df['z_k_2'].to_numpy()
z_k_3 = df['z_k_3'].to_numpy()
x_k_0 = df['x_k_0'].to_numpy()
x_k_1 = df['x_k_1'].to_numpy()
x_k_2 = df['x_k_2'].to_numpy()
x_k_3 = df['x_k_3'].to_numpy()

# Plot
fig1, ax1 = plt.subplots(2)
ax1[0].plot(time, z_k_0)
ax1[0].plot(time, x_k_0)
ax1[0].plot(time, vectorized_integration(x_k_2, dt))
ax1[1].plot(time, z_k_2)
ax1[1].plot(time, x_k_2)
ax1[0].grid()
ax1[1].grid()

fig2, ax2 = plt.subplots(2)
ax2[0].plot(time, z_k_1)
ax2[0].plot(time, x_k_1)
ax2[0].plot(time, vectorized_integration(x_k_3, dt))
ax2[1].plot(time, z_k_3)
ax2[1].plot(time, x_k_3)
ax2[0].grid()
ax2[1].grid()
plt.show()
