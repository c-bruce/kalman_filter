# Script for working out Kalman filter implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GET DATA
df = pd.read_csv('KF_002_mpu6050.csv')

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
x_k_0 = df['x_k_0'].to_numpy()
x_k_1 = df['x_k_1'].to_numpy()

# Plot
fig, ax = plt.subplots(2)
ax[0].plot(time, z_k_0)
ax[0].plot(time, x_k_0)
ax[0].plot(time, vectorized_integration(x_k_1, dt))
ax[1].plot(time, z_k_1)
ax[1].plot(time, x_k_1)
ax[0].grid()
ax[1].grid()
plt.show()
