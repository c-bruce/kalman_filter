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
const float pitch_offset = 2.83; // [deg]
const float pitch_rate_offset = 0.57; // [deg/s]
const float rad2deg = 57.2958;
const int mpu_address = 0x68;
int gyro_cal_int; // Gyroscope calibration int counter
float gyro_x_offset, gyro_y_offset, gyro_z_offset; // Gyroscope roll, pitch, yaw calibration offsets
float AccX, AccY, AccZ, temperature, GyroX, GyroY, GyroZ; // Raw MPU data
float total_vector_acc;
float roll_angle_acc, pitch_angle_acc;
// float roll_angle_acc_trim = -1.3;
// float pitch_angle_acc_trim = 1.8;
float roll_angle, pitch_angle, yaw_angle;
// float roll_level_adjust, pitch_level_adjust;

// Kalman Filter Variables
// Constant ints and floats
const int window = 10;
const float dt = 0.004;
const float fade = 1.2;
// Varying matricies
BLA::Matrix<2> x_k = {0.0, 0.0}; // State [pitch, pitch_rate]
BLA::Matrix<2, 2> P_k = {1.0, 0.0, 0.0, 1.0}; // Covariance matrix
BLA::Matrix<2> z_k = {0.0, 0.0}; // Measurement state [pitch, pitch_rate]
BLA::Matrix<2, 2> R_k = {0.0, 0.0, 0.0, 0.0}; // Measurement covariance matrix
BLA::Matrix<2, 2> K = {0.0, 0.0, 0.0, 0.0}; // Kalman gain matrix
BLA::Matrix<2, 2> inv = {0.0, 0.0, 0.0, 0.0}; // Inverse matrix required to calculate K
// Constant matricies
const BLA::Matrix<2, 2> F_k = {1.0, dt, 0.0, 1.0}; // Kalman gain
const BLA::Matrix<2, 2> B_k = {0.0, 0.0, 0.0, 0.0}; // Control matrix
const BLA::Matrix<2> u_k = {0.0, 0.0}; // Control vector
const BLA::Matrix<2, 2> Q_k = {0.0, 0.0, 0.0, 0.0}; // Untracked noise matrix
const BLA::Matrix<2, 2> H_k = {1.0, 0.0, 0.0, 1.0}; // Transformation matrix
const BLA::Matrix<2, 2> I = {1.0, 0.0, 0.0, 1.0}; // Identity matrix
// Arrays to store rolling data (https://forum.arduino.cc/t/how-to-store-values-in-a-list-or-array/606820/2)
float z_pitch_array[window];
float z_pitch_rate_array[window];
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
// void get_pitch() 
// {
//   z_k.storage(0, 0) = (atan2(AccZ, AccX) * rad2deg) + 90;
// }

// void get_pitch_rate()
// {
//   z_k.storage(0, 1) = GyroX / gyro_scale_factor;
// }

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
  z_k.storage(0, 0) = ((atan2(AccZ, AccX) * rad2deg) + 90) - pitch_offset;
  z_k.storage(0, 1) = (GyroX / gyro_scale_factor) - pitch_rate_offset;
}

void get_R_k()
{
  R_k.storage(0, 0) = get_covariance(z_pitch_array, z_pitch_array, window);
  R_k.storage(0, 1) = get_covariance(z_pitch_array, z_pitch_rate_array, window);
  R_k.storage(1, 0) = get_covariance(z_pitch_rate_array, z_pitch_array, window); // Could be optimized
  R_k.storage(1, 1) = get_covariance(z_pitch_rate_array, z_pitch_rate_array, window);
}

void get_new_sensor_readings()
{
  // Step 1: Get raw mpu data
  readMPUdata();

  // Step 2: Calculate new z_k
  get_z_k();

  // Step 3: Push new z_k into storage arrays at index
  z_pitch_array[index] = z_k.storage(0, 0);
  z_pitch_rate_array[index] = z_k.storage(0, 1);

  // Step 4: Iterate index
  if (index < window) {
    index++;
  }
  else if (index == window) {
    index = 0;
  }
  
  // Step 5: Calculate R_k
  get_R_k();
}

void get_prediction()
{
  x_k = (F_k * x_k) + (B_k * u_k);
  P_k = ((F_k * (P_k * (~F_k))) * pow((pow(fade, 2)), 0.5)) + Q_k;
}

void get_kalman_gain()
{
  inv = BLA::Inverse((H_k * (P_k * (~H_k))) + R_k);
  K = (P_k * ((~H_k) * inv));
}

void get_update()
{
  x_k = x_k + (K * (z_k - (H_k * x_k)));
  P_k = ((I - (K * H_k)) * (P_k * (~(I - (K * H_k))))) + (K * (R_k * (~K)));
}

// // Define Quadcopter Inputs
// int throttle, throttle_mod;
// int reciever_roll_input, reciever_pitch_input, reciever_yaw_input;
// int tuning_trimming, tuning_dir, pb1, pb2, pb3, pb4;
// int pb1_last = 1;
// int pb2_last = 1;
// int pb3_last = 1;
// int pb4_last = 1;

// // Define PID Controllers
// float roll_Kp = 0.7; // Roll p gain
// float roll_Ki = 0.01; // Roll i gain
// float roll_Kd = 0.0; // Roll d gain
// float roll_lim = 400.0; // Roll limit +/-
// float gyro_roll_input, roll_setpoint, roll_error, roll_previous_error, roll_int_error, roll_output; // Input from gyroscope

// float pitch_Kp = roll_Kp; // Pitch p gain
// float pitch_Ki = roll_Ki; // Pitch i gain
// float pitch_Kd = roll_Kd; // Pitch d gain
// float pitch_lim = 400.0; // Pitch limit +/-
// float gyro_pitch_input, pitch_setpoint, pitch_error, pitch_previous_error, pitch_int_error, pitch_output; // Input from gyroscope

// float yaw_Kp = 3.0; // Yaw p gain
// float yaw_Ki = 0.02; // Yaw i gain
// float yaw_Kd = 0.0; // Yaw d gain
// float yaw_lim = 400.0; // Yaw limit +/-
// float gyro_yaw_input, yaw_setpoint, yaw_error, yaw_previous_error, yaw_int_error, yaw_output; // Input from gyroscope

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Radio
////////////////////////////////////////////////////////////////////////////////////////////////////////

// Define data struct for recieving
// Max size of this struct is 32 bytes - NRF24L01 buffer limit
// struct Data_Package
// {
//   byte j1x_VAL;
//   byte j1y_VAL;
//   byte j1b_VAL;
//   byte j2x_VAL;
//   byte j2y_VAL;
//   byte j2b_VAL;
//   byte pot1_VAL;
//   byte pot2_VAL;
//   byte t1_VAL;
//   byte t2_VAL;
//   byte pb1_VAL;
//   byte pb2_VAL;
//   byte pb3_VAL;
//   byte pb4_VAL;
// };

// Data_Package data; // Create a variable with the above structure

// Define RF
// RF24 radio(7, 8); // CE, CSN
// const byte address[6] = "00001";
// unsigned long lastReceiveTime = 0;
// unsigned long currentTime = 0;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Servos
////////////////////////////////////////////////////////////////////////////////////////////////////////

// Define Servos
// bool esc_armed = false;
// int esc_armed_int = 0;
// #define bm1_PIN 2 // Brushless motor 1
// #define bm2_PIN 3 // Brushless motor 2
// #define bm3_PIN 4 // Brushless motor 3
// #define bm4_PIN 5 // Brushless motor 4
// Servo BM1;
// Servo BM2;
// Servo BM3;
// Servo BM4;
// int bm1, bm2, bm3, bm4; // ESC input FL, FR, RL, RR

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////////////////////////////

// void resetData()
// {
//   data.j1x_VAL = 127;
//   data.j1y_VAL = 127;
//   data.j2x_VAL = 127;
//   data.j2y_VAL = 127;
//   data.j1b_VAL = 1;
//   data.j2b_VAL = 1;
//   data.pot1_VAL = 0;
//   data.pot2_VAL = 0;
//   data.t1_VAL = 1;
//   data.t2_VAL = 1;
//   data.pb1_VAL = 1;
//   data.pb2_VAL = 1;
//   data.pb3_VAL = 1;
//   data.pb4_VAL = 1;
// }

// void getRCtransmission()
// {
//   if (radio.available()) // If data is available read it
//   {
//     radio.read(&data, sizeof(Data_Package));
//     lastReceiveTime = millis(); // Get lastRecievedTime
//   }

//   currentTime = millis(); // Get currentTime
  
//   if (currentTime - lastReceiveTime > 1000) // If data hasn't been read for over 1 second reset data
//   {
//     resetData();
//   }
// }

// void setupMPUregisters()
// {
//   // Activate the MPU-6050
//   Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
//   Wire.write(0x6B); // Send the requested starting register
//   Wire.write(0x00); // Set the requested starting register
//   Wire.endTransmission(); // End the transmission
//   //Configure the accelerometer (+/- 8g)
//   Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
//   Wire.write(0x1C); // Send the requested starting register
//   Wire.write(0x10); // Set the requested starting register
//   Wire.endTransmission(); // End the transmission
//   // Configure the gyroscope (500 deg/s full scale)
//   Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
//   Wire.write(0x1B); // Send the requested starting register
//   Wire.write(0x08); // Set the requested starting register
//   Wire.endTransmission(); // End the transmission
// }

// void readMPUdata()
// {
//   // Read raw gyroscope and accelerometer data
//   Wire.beginTransmission(mpu_address); // Start communicating with the MPU-6050
//   Wire.write(0x3B); // Send the requested starting register
//   Wire.endTransmission(); // End the transmission
//   Wire.requestFrom(mpu_address, 14, true); // Request 14 bytes from the MPU-6050
//   AccX = -(Wire.read() << 8 | Wire.read());
//   AccY = -(Wire.read() << 8 | Wire.read());
//   AccZ = (Wire.read() << 8 | Wire.read());
//   temperature = Wire.read() << 8 | Wire.read();
//   GyroX = -(Wire.read() << 8 | Wire.read());
//   GyroY = -(Wire.read() << 8 | Wire.read());
//   GyroZ = (Wire.read() << 8 | Wire.read());
// }

//void getRollPitch(float roll_angle_acc_trim, float pitch_angle_acc_trim)
// void getRollPitch()
// {
//   // Step 1: Get accelerometer and gyroscope data
//   readMPUdata();

//   // Step 2: Subtract gyroscope offsets
//   if (gyro_calibrated == true)
//   {
//     GyroX -= gyro_x_offset;
//     GyroY -= gyro_y_offset;
//     GyroZ -= gyro_z_offset;
//   }

//   // Step 3: Gyroscope angle calculations
//   // 0.0000611 = dt / 65.5, where dt = 0.0004
//   roll_angle += GyroX * 0.0000611;
//   pitch_angle += GyroY * 0.0000611;
  
//   // Step 4: Correct roll and pitch for IMU yawing
//   // 0.000001066 = 0.0000611 * (PI / 180) -> sin function uses radians
//   roll_angle -= pitch_angle * sin(GyroZ * 0.000001066); // If IMU has yawed transfer the pitch angle to the roll angel
//   pitch_angle += roll_angle * sin(GyroZ * 0.000001066); // If IMU has yawed transfer the roll angle to the pitch angel
  
//   // Step 5: Accelerometer angle calculation
//   // 57.296 = 180 / PI -> asin function uses radians
//   total_vector_acc = sqrt((AccX * AccX) + (AccY * AccY) + (AccZ * AccZ)); // Calculate the total accelerometer vector

//   if(abs(AccY) < total_vector_acc) // Prevent asin function producing a NaN
//   {
//     roll_angle_acc = asin((float)AccY/total_vector_acc) * -57.296;
//   }
//   if(abs(AccX) < total_vector_acc) // Prevent asin function producing a NaN
//   {
//     pitch_angle_acc = asin((float)AccX/total_vector_acc) * 57.296;
//   }

//   // Step 6: Correct for roll_angle_acc and pitch_angle_acc offsets (found manually)
//   //roll_angle_acc -= -0.014;
//   //pitch_angle_acc -= 2.344;
//   //roll_angle_acc -= roll_angle_acc_trim;
//   //pitch_angle_acc -= pitch_angle_acc_trim;

//   // Step 7: Set roll and pitch angle depending on if IMU has already started or not
//   if(imu_started) // If the IMU is already started
//   {
//     roll_angle = roll_angle * 0.98 + roll_angle_acc * 0.02; // Correct the drift of the gyro roll angle with the accelerometer roll angle
//     pitch_angle = pitch_angle * 0.98 + pitch_angle_acc * 0.02; // Correct the drift of the gyro pitch angle with the accelerometer pitch angle
//   }
//   else // On first start
//   {
//     roll_angle = roll_angle_acc;
//     pitch_angle = pitch_angle_acc;
//     imu_started = true; // Set the IMU started flag
//   }
//   //roll_level_adjust = roll_angle * 15; // Calculate the roll angle correction
//   //pitch_level_adjust = pitch_angle * 15; // Calculate the pitch angle correction
// }

// void getPIDoutput(float roll_Kp, float roll_Ki, float roll_Kd) // Get PID output
// {
//   pitch_Kp = roll_Kp;
//   pitch_Ki = roll_Ki;
//   pitch_Kd = roll_Kd;
//   // Roll
//   gyro_roll_input = (gyro_roll_input * 0.7) + ((GyroX / 65.5) * 0.3); // 65.5 = 1 deg/s
//   roll_error = gyro_roll_input - roll_setpoint;

//   if (throttle > 1050)
//   {
//     roll_int_error += roll_Ki * roll_error;
//     /*
//     if (roll_int_error > 160) roll_int_error = 160; // Deal with integral wind up
//     else if (roll_int_error < -1 * 160) roll_int_error = -1 * 160;
//     */
//   }
//   else if (throttle < 1050) roll_int_error = 0;

//   roll_output = (roll_Kp * roll_error) + roll_int_error + (roll_Kd * (roll_error - roll_previous_error));
//   if(roll_output > roll_lim) roll_output = roll_lim; // Limit roll output
//   else if(roll_output < roll_lim * -1) roll_output = roll_lim * -1;

//   roll_previous_error = roll_error;

//   // Pitch
//   gyro_pitch_input = (gyro_pitch_input * 0.7) + ((GyroY / 65.5) * 0.3); // 65.5 = 1 deg/s
//   pitch_error = gyro_pitch_input - pitch_setpoint;

//   if (throttle > 1050)
//   {
//     pitch_int_error += pitch_Ki * pitch_error;
//     /*
//     if (pitch_int_error > 160) pitch_int_error = 160; // Deal with integral wind up
//     else if (pitch_int_error < -1 * 160) pitch_int_error = -1 * 160;
//     */
//   }
//   else if (throttle < 1050) pitch_int_error = 0;

//   pitch_output = (pitch_Kp * pitch_error) + pitch_int_error + (pitch_Kd * (pitch_error - pitch_previous_error));
//   if(pitch_output > pitch_lim) pitch_output = pitch_lim; // Limit pitch output
//   else if(pitch_output < pitch_lim * -1) pitch_output = pitch_lim * -1;

//   pitch_previous_error = pitch_error;

//   // Yaw
//   gyro_yaw_input = (gyro_yaw_input * 0.7) + ((GyroZ / 65.5) * 0.3); // 65.5 = 1 deg/s
//   yaw_error = gyro_yaw_input - yaw_setpoint;

//   if (throttle > 1050)
//   {
//     yaw_int_error += yaw_Ki * yaw_error;
//     /*
//     if (yaw_int_error > 160) yaw_int_error = 160; // Deal with integral wind up
//     else if (yaw_int_error < -1 * 160) yaw_int_error = -1 * 160;
//     */
//   }
//   else if (throttle < 1050) yaw_int_error = 0;

//   yaw_output = (yaw_Kp * yaw_error) + yaw_int_error + (yaw_Kd * (yaw_error - yaw_previous_error));
//   if(yaw_output > yaw_lim) yaw_output = yaw_lim; // Limit yaw output
//   else if(yaw_output < yaw_lim * -1) yaw_output = yaw_lim * -1;

//   yaw_previous_error = yaw_error;
// }

void setup()
{
  Serial.begin(57600); // For debugging
  Wire.begin(); // Start the I2C as master

  // Define radio communication
  // radio.begin();
  // radio.openReadingPipe(0, address);
  // radio.setAutoAck(false);
  // radio.setDataRate(RF24_250KBPS);
  // radio.setPALevel(RF24_PA_LOW);
  // radio.startListening();

  // Set default values
  // resetData();

  // MPU setup (reset the sensor through the power management register)
  setupMPUregisters();

  // Servos setup
  // BM1.attach(bm1_PIN, 1000, 2000); // (pin, min pulse width, max pulse width in microseconds)
  // BM2.attach(bm2_PIN, 1000, 2000);
  // BM3.attach(bm3_PIN, 1000, 2000);
  // BM4.attach(bm4_PIN, 1000, 2000);
  // Serial.print("AccX");
  // Serial.print(",");
  // Serial.print("AccY");
  // Serial.print(",");
  // Serial.print("AccZ");
  // Serial.print(",");
  // Serial.print("GyroX");
  // Serial.print(",");
  // Serial.print("GyroY");
  // Serial.print(",");
  // Serial.println("GyroZ");
  Serial.print("z_k_0");
  Serial.print(",");
  Serial.print("z_k_1");
  Serial.print(",");
  Serial.print("x_k_0");
  Serial.print(",");
  Serial.println("x_k_1");

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

  Serial.print(z_k.storage(0, 0));
  Serial.print(",");
  Serial.print(z_k.storage(0, 1));
  Serial.print(",");
  Serial.print(x_k.storage(0, 0));
  Serial.print(",");
  Serial.println(x_k.storage(0, 1));
  // Serial << z_k;
  // Serial.print(',');
  // Serial << x_k;
  // Serial.println();

  // readMPUdata();
  // Serial.print(AccX);
  // Serial.print(",");
  // Serial.print(AccY);
  // Serial.print(",");
  // Serial.print(AccZ);
  // Serial.print(",");
  // Serial.print(GyroX);
  // Serial.print(",");
  // Serial.print(GyroY);
  // Serial.print(",");
  // Serial.println(GyroZ);
  
  // Serial << z_k;
  // Serial.println(';');
  // get_new_sensor_readings();
  // Serial << z_k;
  // Serial.println(';');
  // for(int i = 0; i < window; i++)
  // {
  //   Serial.print(z_pitch_array[i]);
  //   Serial.print(',');
  // }
  // Serial.println(';');


  // loop_timer = micros();
  // // Step 1: Get MPU data
  // getRollPitch(roll_angle_acc_trim, pitch_angle_acc_trim);
  // gyro_roll_input = (gyro_roll_input * 0.7) + ((GyroX / 65.5) * 0.3); // 65.5 = 1 deg/s
  // gyro_pitch_input = (gyro_pitch_input * 0.7) + ((GyroY / 65.5) * 0.3); // 65.5 = 1 deg/s
  // gyro_yaw_input = (gyro_yaw_input * 0.7) + ((GyroZ / 65.5) * 0.3); // 65.5 = 1 deg/s

  // // Step 2: Get transmission from RC controller
  // getRCtransmission();
  // throttle = map(data.pot1_VAL, 0, 255, 1000, 2000);
  // throttle_mod = map(data.j1y_VAL, 0, 255, -200, 200);
  // throttle = throttle + throttle_mod;
  // if (throttle < 1000) throttle = 1000;
  // reciever_roll_input = map(data.j2x_VAL, 0, 255, 1000, 2000);
  // reciever_pitch_input = map(data.j2y_VAL, 0, 255, 2000, 1000);
  // reciever_yaw_input = map(data.j1x_VAL, 0, 255, 1000, 2000);
  // tuning_trimming = data.t1_VAL;
  // tuning_dir = data.t2_VAL;
  // pb1 = data.pb1_VAL;
  // pb2 = data.pb2_VAL;
  // pb3 = data.pb3_VAL;
  // pb4 = data.pb4_VAL;

  // Step 3: PID tuning/roll and pitch trimming
  // if (tuning_trimming == 0) // If tuning is on
  // {
  //   if (tuning_dir == 0) // If tuning direction is in the positive direction
  //   {
  //     // Tuning roll/pitch P gain
  //     if (pb1 != pb1_last)
  //       if (pb1 == 0) roll_Kp += 0.1;
  //     pb1_last = pb1;
  //     // Tuning roll/pitch I gain
  //     if (pb2 != pb2_last)
  //       if (pb2 == 0) roll_Ki += 0.01;
  //     pb2_last = pb2;
  //     // Tuning roll/pitch D gain
  //     if (pb3 != pb3_last)
  //       if (pb3 == 0) roll_Kd += 0.1;
  //     pb3_last = pb3;
  //   }
  //   else if (tuning_dir == 1) // If tuning direction is in the negative direction
  //   {
  //     // Tuning roll/pitch P gain
  //     if (pb1 != pb1_last)
  //       if (pb1 == 0) roll_Kp -= 0.1;
  //     pb1_last = pb1;
  //     // Tuning roll/pitch I gain
  //     if (pb2 != pb2_last)
  //       if (pb2 == 0) roll_Ki -= 0.01;
  //     pb2_last = pb2;
  //     // Tuning roll/pitch D gain
  //     if (pb3 != pb3_last)
  //       if (pb3 == 0) roll_Kd -= 0.1;
  //     pb3_last = pb3;
  //   }
  // }
  // else if (tuning_trimming == 1) // If trimming is on
  // {
  //   // Subtract pitch trim
  //   if (pb1 != pb1_last)
  //     if (pb1 == 0) pitch_angle_acc_trim -= 0.5;
  //   pb1_last = pb1;
  //   // Add pitch trim
  //   if (pb2 != pb2_last)
  //     if (pb2 == 0) pitch_angle_acc_trim += 0.5;
  //   pb2_last = pb2;
  //   // Subtract roll trim
  //   if (pb3 != pb3_last)
  //     if (pb3 == 0) roll_angle_acc_trim -= 0.5;
  //   pb3_last = pb3;
  //   // Add roll trim
  //   if (pb4 != pb4_last)
  //     if (pb4 == 0) roll_angle_acc_trim += 0.5;
  //   pb4_last = pb4;
  // }
  // // Protect against going negative to avoid unwanted behaviour
  // if (roll_Kp < 0) roll_Kp = 0;
  // if (roll_Ki < 0) roll_Ki = 0;
  // if (roll_Kd < 0) roll_Kd = 0;

  // // Step 4: Calculate setpoints
  // roll_setpoint = 0;
  // if(reciever_roll_input > 1520) roll_setpoint = reciever_roll_input - 1520;
  // else if(reciever_roll_input < 1480) roll_setpoint = reciever_roll_input - 1480;
  // roll_setpoint -= roll_level_adjust; // Subtract roll angle correction from the standardized receiver roll input value
  // roll_setpoint /= 3; // Divide roll setpoint for the PID roll controller by 3 to get angles in degrees

  // pitch_setpoint = 0;
  // if(reciever_pitch_input > 1520) pitch_setpoint = reciever_pitch_input - 1520;
  // else if(reciever_pitch_input < 1480) pitch_setpoint = reciever_pitch_input - 1480;
  // pitch_setpoint -= pitch_level_adjust; // Subtract pitch angle correction from the standardized receiver pitch input value
  // pitch_setpoint /= 3; // Divide pitch setpoint for the PID pitch controller by 3 to get angles in degrees

  // yaw_setpoint = 0;
  // if(throttle > 1050) // Do not yaw when turning off the motors.
  // {
  //   if(reciever_yaw_input > 1520) yaw_setpoint = reciever_yaw_input - 1520;
  //   else if(reciever_yaw_input < 1480) yaw_setpoint = reciever_yaw_input - 1480;
  //   yaw_setpoint /= 3; // Divide yaw setpoint for the PID yaw controller by 3 to get angles in degrees
  // }

  // // Step 5: Get PID output
  // getPIDoutput(roll_Kp, roll_Ki, roll_Kd);

  // // Step 6: Calculate BM inputs (arm esc's -> calibrate gyro -> run)
  // if ((esc_armed == false) && (gyro_calibrated == false))
  // {
  //   bm1 = throttle;
  //   bm2 = throttle;
  //   bm3 = throttle;
  //   bm4 = throttle;
  //   if (esc_armed_int < 7500)
  //   {
  //     esc_armed_int += 1;
  //   }
  //   else if (esc_armed_int == 7500)
  //   {
  //     esc_armed = true;
  //     esc_armed_int += 1;
  //   }
  // }
  // else if ((esc_armed == true) && (gyro_calibrated == false))
  // {
  //   bm1 = throttle;
  //   bm2 = throttle;
  //   bm3 = throttle;
  //   bm4 = throttle;
  //   if (gyro_cal_int < 2000)
  //   {
  //     gyro_x_offset += GyroX;
  //     gyro_y_offset += GyroY;
  //     gyro_z_offset += GyroZ;
  //     gyro_cal_int += 1;
  //   }
  //   else if (gyro_cal_int == 2000)
  //   {
  //     gyro_x_offset /= 2000;
  //     gyro_y_offset /= 2000;
  //     gyro_z_offset /= 2000;
  //     gyro_calibrated = true;
  //     gyro_cal_int += 1;
  //   }
  // }
  // else if ((esc_armed == true) && (gyro_calibrated == true))
  // {
  //   // Step 7: Calculate esc input
  //   if (throttle > 1800) throttle = 1800; // We need some room to keep full control at full throttle.
  //   bm1 = throttle - roll_output - pitch_output + yaw_output; // Calculate the pulse for bm1 (front-left - CW)
  //   bm2 = throttle + roll_output - pitch_output - yaw_output; // Calculate the pulse for bm2 (front-right - CCW)
  //   bm3 = throttle - roll_output + pitch_output - yaw_output; // Calculate the pulse for bm3 (rear-left - CCW)
  //   bm4 = throttle + roll_output + pitch_output + yaw_output; // Calculate the pulse for bm4 (rear-right - CW)
  // }
  
  // BM1.writeMicroseconds(bm1);
  // BM2.writeMicroseconds(bm2);
  // BM3.writeMicroseconds(bm3);
  // BM4.writeMicroseconds(bm4);
  
  // Serial.print(roll_angle_acc);
  // Serial.print(", ");
  // Serial.println(pitch_angle_acc);
  
  /*
  Serial.print(bm1);
  Serial.print(", ");
  Serial.print(bm2);
  Serial.print(", ");
  Serial.print(bm3);
  Serial.print(", ");
  Serial.println(bm4);
  */
  /*
  Serial.print(throttle);
  Serial.print(", ");
  Serial.println(reciever_yaw_input);
  */
  /*
  Serial.print(tunintrimming);
  Serial.print(", ");
  Serial.print(tuning_dir);
  Serial.print(", ");
  Serial.print(pb1);
  Serial.print(", ");
  Serial.print(pb2);
  Serial.print(", ");
  Serial.print(pb3);
  Serial.print(", ");
  Serial.print(roll_Kp);
  Serial.print(", ");
  Serial.print(roll_Ki);
  Serial.print(", ");
  Serial.println(roll_Kd);
  */
  //Serial.println(micros() - loop_timer);
  while(micros() - loop_timer < 4000); // Wait until the loop_timer reaches 4000us (250Hz) before starting the next loop
}