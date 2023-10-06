#include <Adafruit_SPIFlash.h>
#include <LSM6DS3.h>
#include <nrf52840.h>
#include <Wire.h>
#include <bluefruit.h>
#include <nrf_saadc.h>
#include <nrf_power.h>

#define LEDR LED_RED
#define LEDG LED_GREEN
#define LEDB LED_BLUE

#define PRESSURE_SENSOR_VCC_PIN D1
#define PRESSURE_SENSOR_SAADC_CHANNEL 0

const unsigned long delayMs = 7;
const unsigned long rate = 1000/delayMs;

struct IMUDataPacket {
  int16_t accel[3];
  int16_t gyro[3];
  uint16_t pressure;
};

const uint8_t LBS_UUID_SERVICE[] =
{
  0x19, 0xB1, 0x00, 0x10, 0xE8, 0xF2, 0x53, 0x7E, 0x4F, 0x6C, 0xD1, 0x04, 0x76, 0x8A, 0x12, 0x14
};

// 19B10013-E8F2-537E-4F6C-D104768A1214
const uint8_t LBS_UUID_CHR_IMU[] =
{
  0x14, 0x12, 0x8A, 0x76, 0x04, 0xD1, 0x6C, 0x4F, 0x7E, 0x53, 0xF2, 0xE8, 0x13, 0x00, 0xB1, 0x19
};

BLEService bleService(LBS_UUID_SERVICE);
BLECharacteristic imuCharacteristic(LBS_UUID_CHR_IMU); //, 0, sizeof(IMUDataPacket), true);

Adafruit_FlashTransport_QSPI flashTransport;
LSM6DS3 myIMU(I2C_MODE, 0x6A); //I2C device address 0x6A

// Disable QSPI flash to save power
void QSPIF_sleep(void)
{
  flashTransport.begin();
  flashTransport.runCommand(0xB9);
  flashTransport.end();
}

void imuISR() {
  // Interrupt triggers for both single and double taps, so we need to check which one it was.
  uint8_t tapSrc;
  status_t status = myIMU.readRegister(&tapSrc, LSM6DS3_ACC_GYRO_TAP_SRC);
  bool wasDoubleTap = (tapSrc >> 4) & 1;
  if (!wasDoubleTap) {
    nrf_power_system_off(NRF_POWER);
  }
}

void setupWakeUpInterrupt()
{
  // Start with LSM6DS3 in disabled to save power
  myIMU.settings.gyroEnabled = 0;
  myIMU.settings.accelEnabled = 0;
  myIMU.begin();

  myIMU.writeRegister(LSM6DS3_ACC_GYRO_CTRL1_XL, 0x30);    // Turn on the accelerometer
                                                           // ODR_XL = 52 Hz, FS_XL = ±2 g
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_TAP_CFG1, 0b10001000);    // Enable interrupts and tap detection on X-axis
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_TAP_THS_6D, 0b10000100);  // Set tap threshold
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_INT_DUR2, 0x10);    // Set Duration, Quiet and Shock time windows
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_WAKE_UP_THS, 0x80); // Single & double-tap enabled (SINGLE_DOUBLE_TAP = 1)
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_MD1_CFG, 0x08);     // Double-tap interrupt driven to INT1 pin
  myIMU.writeRegister(LSM6DS3_ACC_GYRO_CTRL6_G, 0x10);     // High-performance operating mode disabled for accelerometer

  // Set up the sense mechanism to generate the DETECT signal to wake from system_off
  pinMode(PIN_LSM6DS3TR_C_INT1, INPUT_PULLDOWN_SENSE);
  attachInterrupt(digitalPinToInterrupt(PIN_LSM6DS3TR_C_INT1), imuISR, CHANGE);

  return;
}

nrf_saadc_channel_config_t adcConfig = {
  .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
  .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
  .gain       = NRF_SAADC_GAIN4, // GAIN4
  .reference  = NRF_SAADC_REFERENCE_VDD4, // VDD4
  .acq_time   = NRF_SAADC_ACQTIME_20US,
  .mode       = NRF_SAADC_MODE_DIFFERENTIAL,
  .burst      = NRF_SAADC_BURST_ENABLED,
};

void setupPressureSensor() {
  pinMode(PRESSURE_SENSOR_VCC_PIN, OUTPUT);
  digitalWrite(PRESSURE_SENSOR_VCC_PIN, HIGH);

  nrf_saadc_enable(NRF_SAADC);

  nrf_saadc_oversample_set(NRF_SAADC, NRF_SAADC_OVERSAMPLE_32X);
  nrf_saadc_resolution_set(NRF_SAADC, NRF_SAADC_RESOLUTION_12BIT);
  nrf_saadc_channel_input_set(NRF_SAADC, PRESSURE_SENSOR_SAADC_CHANNEL, NRF_SAADC_INPUT_AIN2, NRF_SAADC_INPUT_AIN4);
  nrf_saadc_channel_init(NRF_SAADC, PRESSURE_SENSOR_SAADC_CHANNEL, &adcConfig);

  Serial.println("Calibrating...");
  NRF_SAADC->TASKS_CALIBRATEOFFSET = 1;
  while (NRF_SAADC->EVENTS_CALIBRATEDONE == 0);
  NRF_SAADC->EVENTS_CALIBRATEDONE = 0;
  while (NRF_SAADC->STATUS == (SAADC_STATUS_STATUS_Busy << SAADC_STATUS_STATUS_Pos));

  nrf_saadc_disable(NRF_SAADC);
}

int16_t readPressure() {
  volatile nrf_saadc_value_t result = -1;

  nrf_saadc_enable(NRF_SAADC);
  nrf_saadc_continuous_mode_disable(NRF_SAADC);

  NRF_SAADC->RESULT.MAXCNT = 1;
  NRF_SAADC->RESULT.PTR = (uint32_t)&result;

  // Start the SAADC and wait for the started event.
  NRF_SAADC->TASKS_START = 1;
  while (NRF_SAADC->EVENTS_STARTED == 0);
  NRF_SAADC->EVENTS_STARTED = 0;

  // Do a SAADC sample, will put the result in the configured RAM buffer.
  NRF_SAADC->TASKS_SAMPLE = 1;
  while (NRF_SAADC->EVENTS_END == 0);
  NRF_SAADC->EVENTS_END = 0;

  NRF_SAADC->TASKS_STOP = 1;
  while (NRF_SAADC->EVENTS_STOPPED == 0);
  NRF_SAADC->EVENTS_STOPPED = 0;

  nrf_saadc_disable(NRF_SAADC);

  // Scaling to match previous mbed implementation
  return int16_t((uint32_t(result) * 0xFFFF) / 0x0FFF);
}

void setupImu() {
  myIMU.settings.accelRange = 4; // Can be: 2, 4, 8, 16
  myIMU.settings.gyroRange = 500; // Can be: 125, 245, 500, 1000, 2000
  myIMU.settings.accelSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
  myIMU.settings.gyroSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666
  myIMU.settings.accelBandWidth = 200;
  myIMU.settings.gyroBandWidth = 200;
  myIMU.begin();
}

const float wakeUpThreshold = 200.0f; // 500.0f; // degrees per second
const unsigned long stayAwakeTimeMs = 1000*30; // milliseconds

void run_ble() {
  while (Bluefruit.connected(0)) {
    unsigned long startTime = millis();

    IMUDataPacket packet;
    // This could be optimised by reading all values as a block.
    packet.accel[0] = myIMU.readRawAccelX();
    packet.accel[1] = myIMU.readRawAccelY();
    packet.accel[2] = myIMU.readRawAccelZ();
    packet.gyro[0] = myIMU.readRawGyroX();
    packet.gyro[1] = myIMU.readRawGyroY();
    packet.gyro[2] = myIMU.readRawGyroZ();

    packet.pressure = readPressure();
    
    imuCharacteristic.notify(&packet, sizeof(packet));

    // Inaccurate but usable way to throttle the rate of measurements.
    unsigned long time = millis() - startTime;
    unsigned long waitPeriod = delayMs - time;
    if (waitPeriod > 0 && waitPeriod < 500) { // protection against overflow issues
      delay(waitPeriod);
    }
  }
}

void startAdvertising()
{
  Bluefruit.Advertising.addFlags(BLE_GAP_ADV_FLAGS_LE_ONLY_GENERAL_DISC_MODE);
  Bluefruit.Advertising.addTxPower();
  Bluefruit.Advertising.addService(bleService);

  Bluefruit.ScanResponse.addName();
  
  Bluefruit.Advertising.restartOnDisconnect(true);
  Bluefruit.Advertising.setInterval(128, 488);    // in unit of 0.625 ms
  Bluefruit.Advertising.setFastTimeout(30);      // number of seconds in fast mode
  Bluefruit.Advertising.start(0);                // 0 = Don't stop advertising after n seconds
}

void sleepUntilDoubleTap() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  Serial.println("Setting up interrupt");
  // Setup up double tap interrupt to wake back up
  setupWakeUpInterrupt();

  Serial.println("Entering sleep");
  Serial.flush();

  // Execution should not go beyond this
  nrf_power_system_off(NRF_POWER);
}

void setup() {
  Serial.begin(9600);
  // while (!Serial && millis() < 1000); // Timeout in case serial disconnected.
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, LOW);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);

  QSPIF_sleep();

  Bluefruit.autoConnLed(false);
  Serial.println("Initialise the Bluefruit nRF52 module");
  Bluefruit.configPrphBandwidth(4); // max is BANDWIDTH_MAX = 4
  Serial.print("Begin Bluefruit: ");
  Serial.println(Bluefruit.begin(1, 0));
  Bluefruit.Periph.setConnInterval(6, 6);
  Bluefruit.setName("DPOINT");
  Serial.println("Begin bleService");
  bleService.begin();

  imuCharacteristic.setProperties(CHR_PROPS_READ | CHR_PROPS_NOTIFY);
  imuCharacteristic.setPermission(SECMODE_OPEN, SECMODE_NO_ACCESS);
  imuCharacteristic.setFixedLen(sizeof(IMUDataPacket));
  Serial.println("Begin imuCharacteristic");
  imuCharacteristic.begin();
  IMUDataPacket initialPacket = { 0 };
  imuCharacteristic.write(&initialPacket, sizeof(initialPacket));

  Serial.println("Setup finished");
}

void loop() {
  Serial.print("Starting advertising...");
  startAdvertising();
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, LOW);

  Serial.print("Starting IMU...");
  setupImu();

  unsigned long wakeUpTime = millis();

  while (millis() - wakeUpTime < stayAwakeTimeMs) {
    if (Bluefruit.connected(0)) {
      Serial.println("Connected");
      setupPressureSensor();
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, LOW);
      digitalWrite(LEDB, HIGH);
      run_ble();
      digitalWrite(LEDR, HIGH);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, LOW);
      digitalWrite(PRESSURE_SENSOR_VCC_PIN, LOW);
      wakeUpTime = millis();
    }
    // Don't sleep if USB connected, to make code upload easier.
    if (Serial) wakeUpTime = millis();
    delay(100);
  }
  Serial.println("Stopping advertising");
  Bluefruit.Advertising.stop();
  sleepUntilDoubleTap();
}
