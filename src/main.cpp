#include <Arduino.h>
#include <SPI.h>
#include <nRF24L01.h>
#include <RF24.h>
#include <Wire.h>
#include <Servo.h>
#include <BasicLinearAlgebra.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global Variables
bool imu_started = false;
bool gyro_calibrated = false;
long loop_timer;

// Define MPU Variables
const float acc_scale_factor = 4096; // From MPU6050 datasheet [g]
const float gyro_scale_factor = 65.5; // From MPU6050 datasheet [deg/s]
// const float pitch_offset = 2.83; // [deg]
const float pitch_offset = 0; // [deg]
const float roll_offset = 0.0; // [deg]
const float pitch_rate_offset = 0.57; // [deg/s]
const float roll_rate_offset = 0.0; // [deg/s]
const float rad2deg = 57.2958;
const int mpu_address = 0x68;
int gyro_cal_int; // Gyroscope calibration int counter
float gyro_x_offset, gyro_y_offset, gyro_z_offset; // Gyroscope roll, pitch, yaw calibration offsets
float AccX, AccY, AccZ, temperature, GyroX, GyroY, GyroZ; // Raw MPU data
float total_vector_acc;
float roll_angle_acc, pitch_angle_acc;
float roll_angle, pitch_angle, yaw_angle;

// Kalman Filter Variables
// Constant ints and floats
const int window = 10;
const float dt = 0.004;
const float fade = 1.2;

// Varying matricies
// BLA::Matrix<4, 1, BLA::Array<4,1,double>> x_k = {0.0, 0.0, 0.0, 0.0}; // State [pitch, roll, pitch_rate, roll_rate]
BLA::Matrix<4, 1> x_k = {0.0, 0.0, 0.0, 0.0}; // State [pitch, roll, pitch_rate, roll_rate]
BLA::Matrix<4, 4> P_k = {1.0, 0.0, 0.0, 0.0, 
                         0.0, 1.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}; // Covariance matrix
BLA::Matrix<4, 1> z_k = {0.0, 0.0, 0.0, 0.0}; // Measurement state [pitch, roll, pitch_rate, roll_rate]
BLA::Matrix<4, 4> R_k = {10.0, 0.0, 0.0, 0.0, 
                         0.0, 10.0, 0.0, 0.0,
                         0.0, 0.0, 1.0, 0.0,
                         0.0, 0.0, 0.0, 1.0}; // Measurement covariance matrix
BLA::Matrix<4, 4, BLA::Array<4,4,double>> K = {0.0, 0.0, 0.0, 0.0, 
                                               0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0,
                                               0.0, 0.0, 0.0, 0.0}; // Kalman gain matrix
// BLA::Matrix<4, 4> K = {0.4, 0.0, 0.0, 0.0, 
//                        0.0, 0.4, 0.0, 0.0,
//                        0.0, 0.0, 0.5, 0.0,
//                        0.0, 0.0, 0.0, 0.5}; // Kalman gain matrix
BLA::Matrix<4, 4> inv = {0.0, 0.0, 0.0, 0.0, 
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0}; // Inverse matrix required to calculate K

// Constant matricies
const BLA::Matrix<4, 4> F_k = {1.0, 0.0, dt, 0.0, 
                               0.0, 1.0, 0.0, dt,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0}; // State transition matrix
const BLA::Matrix<4, 4> B_k = {0.0, 0.0, 0.0, 0.0, 
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0,
                               0.0, 0.0, 0.0, 0.0}; // Control matrix
const BLA::Matrix<4, 1> u_k = {0.0, 0.0, 0.0, 0.0}; // Control vector
const BLA::Matrix<4, 4> Q_k = {0.0001, 0.0, 0.0, 0.0, 
                               0.0, 0.0001, 0.0, 0.0,
                               0.0, 0.0, 0.0001, 0.0,
                               0.0, 0.0, 0.0, 0.0001};  // Untracked noise matrix
// const BLA::Matrix<4, 4> Q_k = {3.2e-10, 0.0, 1.6e-7, 0.0, 
//                                0.0, 3.2e-10, 0.0, 1.6e-7,
//                                1.6e-7, 0.0, 8.0e-5, 0.0,
//                                0.0, 1.6e-7, 0.0, 8.0e-5};  // Untracked noise matrix
const BLA::Matrix<4, 4> H_k = {1.0, 0.0, 0.0, 0.0, 
                               0.0, 1.0, 0.0, 0.0,
                               0.0, 0.0, 1.0, 0.0,
                               0.0, 0.0, 0.0, 1.0}; // Transformation matrix
const BLA::Matrix<4, 4> I = {1.0, 0.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0}; // Identity matrix

// Arrays to store state data
float z_pitch_array[window];
float z_roll_array[window];
float z_pitch_rate_array[window];
float z_roll_rate_array[window];
byte index = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// MPU
////////////////////////////////////////////////////////////////////////////////////////////////////////

void setupMPUregisters()
{
  // Activate the MPU-6050
  Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
  Wire.write(0x6B); // Send the requested starting register
  Wire.write(0x00); // Set the requested starting register
  Wire.endTransmission(); // End the transmission

  //Configure the accelerometer (+/- 8g)
  Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
  Wire.write(0x1C); // Send the requested starting register
  Wire.write(0x10); // Set the requested starting register
  Wire.endTransmission(); // End the transmission

  // Configure the gyroscope (500 deg/s full scale)
  Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
  Wire.write(0x1B); // Send the requested starting register
  Wire.write(0x08); // Set the requested starting register
  Wire.endTransmission(); // End the transmission
}

void readMPUdata()
{
  // Read raw gyroscope and accelerometer data
  Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
  Wire.write(0x3B); // Send the requested starting register
  Wire.endTransmission(); // End the transmission
  Wire.requestFrom(mpu_address, 14, true); // Request 14 bytes from the MPU-6050
  AccX = -(Wire.read() << 8 | Wire.read());
  AccY = -(Wire.read() << 8 | Wire.read());
  AccZ = (Wire.read() << 8 | Wire.read());
  temperature = Wire.read() << 8 | Wire.read();
  GyroX = -(Wire.read() << 8 | Wire.read());
  GyroY = -(Wire.read() << 8 | Wire.read());
  GyroZ = (Wire.read() << 8 | Wire.read());
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kalman Filter
////////////////////////////////////////////////////////////////////////////////////////////////////////

float get_covariance(const float * a, const float * b, const int n)
{
  // Calculate averages
  float average_a = 0.0;
  float average_b = 0.0;
  for (int i = 0; i < n; i++)
  {
    average_a += a[i];
    average_b += b[i];
  }
  average_a /= n;
  average_b /= n;

  // Calculate covariance
  float covariance = 0.0;
  for (int i = 0; i < n; i++)
  {
    covariance += (a[i] - average_a) * (b[i] - average_b);
  }
  covariance /= n;

  return covariance;
}

void get_z_k()
{
  total_vector_acc = sqrt((AccX * AccX) + (AccY * AccY) + (AccZ * AccZ)); // Calculate the total accelerometer vector

  if(abs(AccY) < total_vector_acc) // Prevent asin function producing a NaN
  {
    z_k.storage(0, 1) = asin((float)AccY/total_vector_acc) * -rad2deg; // roll
  }
  if(abs(AccX) < total_vector_acc) // Prevent asin function producing a NaN
  {
    z_k.storage(0, 0) = asin((float)AccX/total_vector_acc) * rad2deg; // pitch
  }
  // z_k.storage(0, 0) = ((atan2(AccZ, AccX) * rad2deg) + 90) - pitch_offset;
  // z_k.storage(0, 1) = -(((atan2(AccZ, AccY) * rad2deg)+ 90 ) - roll_offset);
  // z_k.storage(0, 0) = ((atan2(AccZ, AccX) * rad2deg) + 90) - pitch_offset;
  // z_k.storage(0, 1) = ((atan2(AccZ, AccY) * rad2deg) + 90 ) - roll_offset;
  z_k.storage(0, 2) = (GyroY / gyro_scale_factor) - pitch_rate_offset;
  z_k.storage(0, 3) = (GyroX / gyro_scale_factor) - roll_rate_offset;
}

void get_R_k()
{
  R_k.storage(0, 0) = get_covariance(z_pitch_array, z_pitch_array, window);
  R_k.storage(0, 1) = get_covariance(z_pitch_array, z_roll_array, window);
  R_k.storage(0, 2) = get_covariance(z_pitch_array, z_pitch_rate_array, window);
  R_k.storage(0, 3) = get_covariance(z_pitch_array, z_roll_rate_array, window);
  //
  R_k.storage(1, 0) = get_covariance(z_roll_array, z_pitch_array, window);
  R_k.storage(1, 1) = get_covariance(z_roll_array, z_roll_array, window);
  R_k.storage(1, 2) = get_covariance(z_roll_array, z_pitch_rate_array, window);
  R_k.storage(1, 3) = get_covariance(z_roll_array, z_roll_rate_array, window);
  //
  R_k.storage(2, 0) = get_covariance(z_pitch_rate_array, z_pitch_array, window);
  R_k.storage(2, 1) = get_covariance(z_pitch_rate_array, z_roll_array, window);
  R_k.storage(2, 2) = get_covariance(z_pitch_rate_array, z_pitch_rate_array, window);
  R_k.storage(2, 3) = get_covariance(z_pitch_rate_array, z_roll_rate_array, window);
  //
  R_k.storage(3, 0) = get_covariance(z_roll_rate_array, z_pitch_array, window);
  R_k.storage(3, 1) = get_covariance(z_roll_rate_array, z_roll_array, window);
  R_k.storage(3, 2) = get_covariance(z_roll_rate_array, z_pitch_rate_array, window);
  R_k.storage(3, 3) = get_covariance(z_roll_rate_array, z_roll_rate_array, window);
}

void get_new_sensor_readings()
{
  // Step 1: Get raw mpu data
  readMPUdata();

  // Step 2: Calculate new z_k
  get_z_k();

  // Step 3: Push new z_k into storage arrays at index
  z_pitch_array[index] = z_k.storage(0, 0);
  z_roll_array[index] = z_k.storage(0, 1);
  z_pitch_rate_array[index] = z_k.storage(0, 2);
  z_roll_rate_array[index] = z_k.storage(0, 3);

  // Step 4: Iterate index
  if (index < window) {
    index++;
  }
  else if (index == window) {
    index = 0;
  }
  
  // Step 5: Calculate R_k
  // get_R_k();
}

void get_prediction()
{
  x_k = (F_k * x_k) + (B_k * u_k);
  P_k = ((F_k * (P_k * (~F_k))) * pow((pow(fade, 2)), 0.5)) + (Q_k * 1000);
  // P_k = ((F_k * (P_k * (~F_k))) * 1.0) + Q_k;
}

void get_kalman_gain()
{
  inv = BLA::Inverse((H_k * (P_k * (~H_k))) + R_k);
  K = ((P_k * (~H_k)) * inv);
}

void get_update()
{
  x_k = x_k + (K * (z_k - (H_k * x_k)));
  P_k = (((I - (K * H_k)) * P_k) * (~(I - (K * H_k)))) + ((K * R_k) * (~K));
}

void setup()
{
  Serial.begin(57600); // For debugging
  Wire.begin(); // Start the I2C as master

  // MPU setup (reset the sensor through the power management register)
  setupMPUregisters();

  // Print header
  // Serial.print("phi");
  // Serial.print(",");
  // Serial.print("theta");
  // Serial.print(",");
  // Serial.println("psi");
  Serial.print("z_k_0");
  Serial.print(",");
  Serial.print("z_k_1");
  Serial.print(",");
  Serial.print("z_k_2");
  Serial.print(",");
  Serial.print("z_k_3");
  Serial.print(",");
  Serial.print("x_k_0");
  Serial.print(",");
  Serial.print("x_k_1");
  Serial.print(",");
  Serial.print("x_k_2");
  Serial.print(",");
  Serial.println("x_k_3");

  loop_timer = micros(); // Reset the loop timer
}

void loop()
{
  // Prediction step
  get_prediction();
  // Update step
  get_new_sensor_readings();
  get_kalman_gain();
  get_update();

  // Serial.print(x_k.storage(0, 1)); // phi
  // Serial.print(",");
  // Serial.print(x_k.storage(0, 0)); // theta
  // Serial.print(",");
  // Serial.println("0"); // psi
  Serial.print(z_k.storage(0, 0));
  Serial.print(",");
  Serial.print(z_k.storage(0, 1));
  Serial.print(",");
  Serial.print(z_k.storage(0, 2));
  Serial.print(",");
  Serial.print(z_k.storage(0, 3));
  Serial.print(",");
  Serial.print(x_k.storage(0, 0));
  Serial.print(",");
  Serial.print(x_k.storage(0, 1));
  Serial.print(",");
  Serial.print(x_k.storage(0, 2));
  Serial.print(",");
  Serial.println(x_k.storage(0, 3));
  
  // Serial.println(micros() - loop_timer);
  while(micros() - loop_timer < 4000); // Wait until the loop_timer reaches 4000us (250Hz) before starting the next loop
}