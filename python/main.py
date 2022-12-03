# Script for working out Kalman filter implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# GET DATA
df = pd.read_csv('002_mpu6050_RAW.csv')

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

def vectorized_integration(input_vector, dt):
    output_vector = [0]
    for i in range(len(input_vector)):
        output_vector.append(output_vector[i] + input_vector[i] * dt)
    return np.array(output_vector[1:])

def get_angle_from_acc(a, b):
    return np.degrees(np.arctan2(b, a) + 1.5)

# def get_alpha_from_acc(AccX, AccY, AccZ):
#     total_vector_acc = np.sqrt(sum([AccX**2, AccY**2, AccZ**2]))
#     print(total_vector_acc)
#     return np.arcsin(AccY/total_vector_acc) * -57.296

def get_angle_from_complimentary_filter(a, b, input_vector, dt):
    output_vector = []
    alpha_acc = get_angle_from_acc(a, b)
    for i in range(len(input_vector)):
        alpha_CF = ((input_vector[i] * dt) * 0.98) + ((alpha_acc[i]) * 0.02)
        output_vector.append(alpha_CF)
    return np.array(output_vector)

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
AccX_SU, AccY_SU, AccZ_SU, GyroX_SU, GyroY_SU, GyroZ_SU = get_raw_in_sensor_units(df)
AccX_PU, AccY_PU, AccZ_PU, GyroX_PU, GyroY_PU, GyroZ_PU = get_raw_in_physical_units(df, acc_scale_factor, gyro_scale_factor)

AngX_PU = vectorized_integration(GyroX_PU, dt)
AngY_PU = vectorized_integration(GyroY_PU, dt)
AngZ_PU = vectorized_integration(GyroZ_PU, dt)

# alpha = AngY_PU
alpha = get_angle_from_acc(AccX_PU, AccZ_PU)
# alpha = get_angle_from_complimentary_filter(AccX_PU, AccZ_PU, GyroY_PU, dt)
alpha_CF = get_angle_from_complimentary_filter(AccX_PU, AccZ_PU, GyroY_PU, dt)
# alpha = get_alpha_from_acc(AccX_PU, AccY_PU, AccZ_PU)

# plt.plot(time, AccX_PU)
# plt.plot(time, AccY_PU)
# plt.plot(time, AccZ_PU)

# plt.plot(time, GyroX_PU)
# plt.plot(time, GyroY_PU)
# plt.plot(time, GyroZ_PU)

# plt.plot(time, AngX_PU)
# plt.plot(time, AngY_PU)
# plt.plot(time, AngZ_PU)
# plt.show()

# KALMAN FILTER
# Constant matricies
F_k = np.array([[1, dt],
                [0, 1]]) # Prediction matrix
B_k = np.array([[0, 0],
                [0, 0]]) # Control matrix
u_k = np.array([0, 0]) # Control vector
Q_k = np.array([[0, 0],
                [0, 0]]) # Untracked noise
H_k = np.array([[1, 0],
                [0, 1]]) # Transformation matrix
# Step 0: Get initial state and covariance matrix
# Use first 0.04 seconds of raw alpha and GyroY_PU data
# window = int(0.04 / dt) # Size of window for getting sensor readings over
# x_k = np.array([np.mean(alpha[0:window]), np.mean(GyroY_PU[0:window])])
# P_k = np.array([[variance(alpha[0:window]), covariance(alpha[0:window], GyroY_PU[0:window])],
#                 [covariance(GyroY_PU[0:window], alpha[0:window]), variance(GyroY_PU[0:window])]])
#
# Step 1: Prediction step
# Get prediction
# x_k = np.dot(F_k, x_k) + np.dot(B_k, u_k)
# P_k = np.dot(F_k, np.dot(P_k, F_k.T)) + Q_k
# # Print prediction
# print(f"Predicted x_k: {x_k}")
# print(f"Predicted P_k: {P_k}")

# # Step 2: Update step
# # Get new sensor readings
# z_k = np.array([np.mean(alpha[1:window+1]), np.mean(GyroY_PU[1:window+1])])
# R_k = np.array([[variance(alpha[1:window+1]), covariance(alpha[1:window+1], GyroY_PU[1:window+1])],
#                 [covariance(GyroY_PU[1:window+1], alpha[1:window+1]), variance(GyroY_PU[1:window+1])]])
# # Calculate Kalman gain
# K = np.dot(P_k, np.dot(H_k.T, np.linalg.inv(np.dot(H_k, np.dot(P_k, H_k)) + R_k)))
# # Calculate updated x_k and P_k
# x_k = x_k + np.dot(K, (z_k - np.dot(H_k, x_k)))
# P_k = P_k - np.dot(K, np.dot(H_k, P_k))

# print(f"Updated x_k: {x_k}")
# print(f"Updated P_k: {P_k}")

# Put it all in a loop...
def get_prediction(F_k, x_k, B_k, u_k, P_k, Q_k):
    x_k = np.dot(F_k, x_k) + np.dot(B_k, u_k)
    P_k = np.dot(F_k, np.dot(P_k, F_k.T)) + Q_k
    return x_k, P_k

def get_new_sensor_readings(start, end):
    z_k = np.array([np.mean(alpha[start:end]), np.mean(GyroY_PU[start:end])])
    R_k = np.array([[variance(alpha[start:end]), covariance(alpha[start:end], GyroY_PU[start:end])],
                    [covariance(GyroY_PU[start:end], alpha[start:end]), variance(GyroY_PU[start:end])]])
    return z_k, R_k

def get_kalman_gain(P_k, H_k, R_k):
    return np.dot(P_k, np.dot(H_k.T, np.linalg.inv(np.dot(H_k, np.dot(P_k, H_k.T)) + R_k)))

def get_update(x_k, K, z_k, H_k, P_k):
    x_k = x_k + np.dot(K, (z_k - np.dot(H_k, x_k)))
    P_k = P_k - np.dot(K, np.dot(H_k, P_k))
    return x_k, P_k

# Step 0: Get initial state and covariance matrix
# Use first 0.04 seconds of raw alpha and GyroY_PU data
window = 10#int(0.04 / dt) # Size of window for getting sensor readings over
x_k, P_k = get_new_sensor_readings(0, window)
alpha_KF = [x_k[0]]
alpha_d_KF = [x_k[1]]
# Start of loop
for i in range(window + 1, len(time)):
    # Step 1: Prediction step
    # Calculate prediction
    x_k, P_k = get_prediction(F_k, x_k, B_k, u_k, P_k, Q_k)
    alpha_KF.append(x_k[0])
    alpha_d_KF.append(x_k[1])
    
    # Step 2: Update step
    # Get new sensor readings
    z_k, R_k = get_new_sensor_readings(i - window, i)
    # Calculate Kalman gain
    K = get_kalman_gain(P_k, H_k, R_k)
    # Calculate update
    x_k, P_k = get_update(x_k, K, z_k, H_k, P_k)
    # Store updated state
    # alpha_KF.append(x_k[0])
    # alpha_d_KF.append(x_k[1])

alpha_KF = np.array(alpha_KF)
alpha_d_KF = np.array(alpha_d_KF)
# Plot
plt.plot(time, alpha)
plt.plot(time, alpha_CF*15)
plt.plot(time, AngY_PU)
plt.plot(time[window:], alpha_KF)
plt.show()

# plt.plot(time, alpha)
# plt.plot(time, GyroY_PU)
# plt.plot(time[window:], alpha_d_KF)
# plt.show()