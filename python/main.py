# Script for working out Kalman filter implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GET DATA
df = pd.read_csv('001_mpu6050_RAW.csv')

# CONSTANTS
dt = 0.004 # [s]
freq = 1/dt # [Hz]
g = 9.81 # [m/s**2]
acc_scale_factor = 4096 # From MPU6050 datasheet [g]
gyro_scale_factor = 65.5 # From MPU6050 datasheet [deg/s]

def get_time(df, dt, freq):
    return np.arange(0, len(df)/freq, dt)

def get_raw_in_sensor_units(df):
    AccX_SU = df['AccX'].to_numpy()
    AccY_SU = df['AccY'].to_numpy()
    AccZ_SU = df['AccZ'].to_numpy()
    GyroX_SU = df['GyroX'].to_numpy()
    GyroY_SU = df['GyroY'].to_numpy()
    GyroZ_SU = df['GyroZ'].to_numpy()
    return AccX_SU, AccY_SU, AccZ_SU, GyroX_SU, GyroY_SU, GyroZ_SU

def get_raw_in_physical_units(df, acc_scale_factor, gyro_scale_factor):
    AccX_PU = df['AccX'].to_numpy() / acc_scale_factor
    AccY_PU = df['AccY'].to_numpy() / acc_scale_factor
    AccZ_PU = df['AccZ'].to_numpy() / acc_scale_factor
    GyroX_PU = df['GyroX'].to_numpy() / gyro_scale_factor
    GyroY_PU = df['GyroY'].to_numpy() / gyro_scale_factor
    GyroZ_PU = df['GyroZ'].to_numpy() / gyro_scale_factor
    return AccX_PU, AccY_PU, AccZ_PU, GyroX_PU, GyroY_PU, GyroZ_PU

def vectorized_integration(dt, input_vector):
    output_vector = [0]
    for i in range(len(input_vector)):
        output_vector.append(output_vector[i] + input_vector[i] * dt)
    return np.array(output_vector[1:])

def variance(input_vector):
    mean = np.mean(input_vector)
    n = len(input_vector)
    return sum((input_vector - mean)**2) / (n - 1)

def covariance(input_vector1, input_vector2):
    mean1 = np.mean(input_vector1)
    mean2 = np.mean(input_vector2)
    n = len(input_vector1)
    return sum((input_vector1 - mean1) * (input_vector2 - mean2)) / (n - 1)

time = get_time(df, dt, freq)
AccX_PU, AccY_PU, AccZ_PU, GyroX_PU, GyroY_PU, GyroZ_PU = get_raw_in_physical_units(df, acc_scale_factor, gyro_scale_factor)

AngX_PU = vectorized_integration(dt, GyroX_PU)
AngY_PU = vectorized_integration(dt, GyroY_PU)
AngZ_PU = vectorized_integration(dt, GyroZ_PU)

# plt.plot(time, AccX_PU)
# plt.plot(time, AccY_PU)
# plt.plot(time, AccZ_PU)

# plt.plot(time, GyroX_PU)
plt.plot(time, GyroY_PU)
# plt.plot(time, GyroZ_PU)

# plt.plot(time, AngX_PU)
plt.plot(time, AngY_PU)
# plt.plot(time, AngZ_PU)
plt.show()

# KALMAN FILTER
# Step 0: Get initial state and covariance matrix
# Use first 0.04 seconds of raw AccY_PU and GyroY_PU data
window = int(0.04 / dt) # Size of window for getting sensor readings over
x_k = np.array([np.mean(AngY_PU[0:window]), np.mean(GyroY_PU[0:window])])
P_k = np.array([[variance(AngY_PU[0:window]), covariance(AngY_PU[0:window], GyroY_PU[0:window])],
                [covariance(GyroY_PU[0:window], AngY_PU[0:window]), variance(GyroY_PU[0:window])]])

# Step 1: Prediction step
F_k = np.array([[1, dt],
                [0, 1]]) # Prediction matrix
B_k = np.array([[0, 0],
                [0, 0]]) # Control matrix
u_k = np.array([0, 0]) # Control vector
Q_k = np.array([[0, 0],
                [0, 0]]) # Untracked noise
# Get prediction
x_k = np.dot(F_k, x_k) + np.dot(B_k, u_k)
P_k = np.dot(F_k, np.dot(P_k, F_k.T)) + Q_k

# Step 2: Update step
# Get new sensor readings
