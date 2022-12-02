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
    return output_vector[1:]

time = get_time(df, dt, freq)
AccX_PU, AccY_PU, AccZ_PU, GyroX_PU, GyroY_PU, GyroZ_PU = get_raw_in_physical_units(df, acc_scale_factor, gyro_scale_factor)

AngX_PU = vectorized_integration(dt, GyroX_PU)
AngY_PU = vectorized_integration(dt, GyroY_PU)
AngZ_PU = vectorized_integration(dt, GyroZ_PU)

# plt.plot(time, AccX_PU)
# plt.plot(time, AccY_PU)
# plt.plot(time, AccZ_PU)

plt.plot(time, GyroX_PU)
#plt.plot(time, GyroY_PU)
#plt.plot(time, GyroZ_PU)

plt.plot(time, AngX_PU)
#plt.plot(time, AngY_PU)
#plt.plot(time, AngZ_PU)
plt.show()