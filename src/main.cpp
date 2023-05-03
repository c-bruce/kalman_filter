#include <Arduino.h>
#include <SPI.h>
#include <Wire.h>
#include <BasicLinearAlgebra.h>

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////////////////////////////

// Global Variables
long loop_timer;

// Define MPU Variables
const float acc_scale_factor = 4096; // From MPU6050 datasheet [g]
const float gyro_scale_factor = 65.5; // From MPU6050 datasheet [deg/s]
const float pitch_offset = 0.0; // [deg]
const float roll_offset = 0.0; // [deg]
const float pitch_rate_offset = 0.0; // [deg/s]
const float roll_rate_offset = 0.0; // [deg/s]
const float rad2deg = 57.2958;
const int mpu_address = 0x68;
float gyro_x_offset, gyro_y_offset, gyro_z_offset; // Gyroscope roll, pitch, yaw calibration offsets
float AccX, AccY, AccZ, temperature, GyroX, GyroY, GyroZ; // Raw MPU data
float total_vector_acc;

// Kalman Filter Variables
// Constant floats
const float dt = 0.004;
const float fade = 1.0;

// Continuous Adjustment Variables
int count = 0;
BLA::Matrix<1, 1> epsilon = {0.0}; // Normalized square of the residual
float epsilon_max = 5.0;
float Q_scale_factor = 1000.0;

// Varying matricies
BLA::Matrix<4, 1> x = {0.0, 0.0, 0.0, 0.0}; // State [roll, pitch, roll_rate, pitch_rate]
BLA::Matrix<4, 1> z = {0.0, 0.0, 0.0, 0.0}; // Measurement state [roll, pitch, roll_rate, pitch_rate]
BLA::Matrix<4, 1> y = {0.0, 0.0, 0.0, 0.0}; // Residule [roll, pitch, roll_rate, pitch_rate]
BLA::Matrix<4, 4> P = {1.0, 0.0, 0.0, 0.0, 
                       0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0,
                       0.0, 0.0, 0.0, 1.0}; // Covariance matrix
BLA::Matrix<4, 4> Q = {pow(dt, 4) / 4, 0.0, 0.0, 0.0, 
                       0.0, pow(dt, 4) / 4, 0.0, 0.0,
                       0.0, 0.0, pow(dt, 2), 0.0,
                       0.0, 0.0, 0.0, pow(dt, 2)};  // Untracked noise matrix
BLA::Matrix<4, 4> K = {0.0, 0.0, 0.0, 0.0, 
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0,
                       0.0, 0.0, 0.0, 0.0}; // Kalman gain matrix
BLA::Matrix<4, 4> inv = {0.0, 0.0, 0.0, 0.0, 
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0, 0.0}; // Inverse matrix required to calculate K

// Constant matricies
const BLA::Matrix<4, 1> u = {0.0, 0.0, 0.0, 0.0}; // Control vector
const BLA::Matrix<4, 4> R = {10.0, 0.0, 0.0, 0.0, 
                             0.0, 10.0, 0.0, 0.0,
                             0.0, 0.0, 10.0, 0.0,
                             0.0, 0.0, 0.0, 10.0}; // Measurement covariance matrix
const BLA::Matrix<4, 4> F = {1.0, 0.0, dt, 0.0, 
                             0.0, 1.0, 0.0, dt,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0}; // State transition matrix
const BLA::Matrix<4, 4> B = {0.0, 0.0, 0.0, 0.0, 
                             0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0,
                             0.0, 0.0, 0.0, 0.0}; // Control matrix
const BLA::Matrix<4, 4> H = {1.0, 0.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0}; // Observation matrix
const BLA::Matrix<4, 4> I = {1.0, 0.0, 0.0, 0.0, 
                             0.0, 1.0, 0.0, 0.0,
                             0.0, 0.0, 1.0, 0.0,
                             0.0, 0.0, 0.0, 1.0}; // Identity matrix

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
  // AccX = Wire.read() << 8 | Wire.read();
  // AccY = Wire.read() << 8 | Wire.read();
  // AccZ = Wire.read() << 8 | Wire.read();
  // temperature = Wire.read() << 8 | Wire.read();
  // GyroX = Wire.read() << 8 | Wire.read();
  // GyroY = Wire.read() << 8 | Wire.read();
  // GyroZ = Wire.read() << 8 | Wire.read();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Kalman Filter
////////////////////////////////////////////////////////////////////////////////////////////////////////

void get_z()
{
  total_vector_acc = sqrt((AccX * AccX) + (AccY * AccY) + (AccZ * AccZ)); // Calculate the total accelerometer vector

  if(abs(AccY) < total_vector_acc) // Prevent asin function producing a NaN
  {
    z.storage(0, 0) = asin((float)AccY/total_vector_acc) * -rad2deg; // roll
  }
  if(abs(AccX) < total_vector_acc) // Prevent asin function producing a NaN
  {
    z.storage(0, 1) = asin((float)AccX/total_vector_acc) * rad2deg; // pitch
  }
  z.storage(0, 2) = (GyroX / gyro_scale_factor) - roll_rate_offset; // roll rate
  z.storage(0, 3) = (GyroY / gyro_scale_factor) - pitch_rate_offset; // pitch rate
}

void get_new_sensor_readings()
{
  readMPUdata();

  get_z();
}

void get_prediction()
{
  x = (F * x) + (B * u);
  P = ((F * (P * (~F))) * pow((pow(fade, 2)), 0.5)) + Q;
}

void get_kalman_gain()
{
  inv = BLA::Inverse((H * (P * (~H))) + R);
  K = ((P * (~H)) * inv);
}

void get_update()
{
  x = x + (K * (z - (H * x)));
  P = (((I - (K * H)) * P) * (~(I - (K * H)))) + ((K * R) * (~K));
}

void get_residual()
{
  y = z - x;
}

void get_epsilon()
{
  epsilon = (~y * (inv * y));
}

void scale_Q()
{
  if (epsilon.storage(0, 0) > epsilon_max)
  {
    Q *= Q_scale_factor;
    count += 1;
  }
  else if (count > 0)
  {
    Q /= Q_scale_factor;
    count -= 1;
  }
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
  Serial.print("z_0");
  Serial.print(",");
  Serial.print("z_1");
  Serial.print(",");
  Serial.print("z_2");
  Serial.print(",");
  Serial.print("z_3");
  Serial.print(",");
  Serial.print("x_0");
  Serial.print(",");
  Serial.print("x_1");
  Serial.print(",");
  Serial.print("x_2");
  Serial.print(",");
  Serial.print("x_3");
  Serial.print(",");
  Serial.print("y_0");
  Serial.print(",");
  Serial.print("y_1");
  Serial.print(",");
  Serial.print("y_2");
  Serial.print(",");
  Serial.print("y_3");
  Serial.print(",");
  Serial.println("epsilon");

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

  // Calculate residual
  get_residual();

  // Continuous adjustment step
  get_epsilon();
  scale_Q();

  // Serial.print(x.storage(0, 1)); // phi
  // Serial.print(",");
  // Serial.print(x.storage(0, 0)); // theta
  // Serial.print(",");
  // Serial.println("0"); // psi
  Serial.print(z.storage(0, 0));
  Serial.print(",");
  Serial.print(z.storage(0, 1));
  Serial.print(",");
  Serial.print(z.storage(0, 2));
  Serial.print(",");
  Serial.print(z.storage(0, 3));
  Serial.print(",");
  Serial.print(x.storage(0, 0));
  Serial.print(",");
  Serial.print(x.storage(0, 1));
  Serial.print(",");
  Serial.print(x.storage(0, 2));
  Serial.print(",");
  Serial.print(x.storage(0, 3));
  Serial.print(",");
  Serial.print(y.storage(0, 0));
  Serial.print(",");
  Serial.print(y.storage(0, 1));
  Serial.print(",");
  Serial.print(y.storage(0, 2));
  Serial.print(",");
  Serial.print(y.storage(0, 3));
  Serial.print(",");
  Serial.println(epsilon.storage(0, 0));
  
  while(micros() - loop_timer < 4000); // Wait until the loop_timer reaches 4000us (250Hz) before starting the next loop
}