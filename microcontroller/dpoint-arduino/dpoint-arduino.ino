#include <LSM6DS3.h>
#include <ArduinoBLE.h>
#include "mbed.h"
#include "Wire.h"

mbed::AnalogIn pressureIn(AIN4);
analogin_config_t adcConfig = {
    .resistor_p = NRF_SAADC_RESISTOR_DISABLED,
    .resistor_n = NRF_SAADC_RESISTOR_DISABLED,
    .gain       = NRF_SAADC_GAIN4,
    .reference  = NRF_SAADC_REFERENCE_VDD4,
    .acq_time   = NRF_SAADC_ACQTIME_10US,
    .mode       = NRF_SAADC_MODE_DIFFERENTIAL,
    .burst      = NRF_SAADC_BURST_ENABLED,
    .pin_p      = NRF_SAADC_INPUT_AIN5, // Should be AIN2, something wrong in seeed files.
    .pin_n      = NRF_SAADC_INPUT_AIN4,
};
#define PRESSURE_SENSOR_VCC_PIN D1

const unsigned long delayMs = 7;
const unsigned long rate = 1000/delayMs;

struct IMUDataPacket {
  int16_t accel[3];
  int16_t gyro[3];
  uint16_t pressure;
};

BLEService dpointService("19B10010-E8F2-537E-4F6C-D104768A1214");

BLEDescriptor imuDescriptor("2901", "IMU");
BLETypedCharacteristic<IMUDataPacket> imuCharacteristic("19B10013-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

LSM6DS3 myIMU(I2C_MODE, 0x6A); //I2C device address 0x6A

void setupPressureSensor() {
  pinMode(PRESSURE_SENSOR_VCC_PIN, OUTPUT);
  digitalWrite(PRESSURE_SENSOR_VCC_PIN, HIGH);
  nrf_saadc_oversample_set(NRF_SAADC_OVERSAMPLE_16X);
  nrf_saadc_resolution_set(NRF_SAADC_RESOLUTION_12BIT);
  pressureIn.configure(adcConfig);
}

void blink_error() {
  for (int i = 0; i < 11; i++) {
    digitalWrite(LEDR, (i % 2 == 0) ? HIGH : LOW);
    delay(200);
  }
}

bool start_ble() {
  if (!BLE.begin()) {
    return false;
  }

  myIMU.settings.accelRange = 4; // Can be: 2, 4, 8, 16
  myIMU.settings.gyroRange = 500; // Can be: 125, 245, 500, 1000, 2000
  myIMU.settings.accelSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
  myIMU.settings.gyroSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666
  myIMU.settings.accelBandWidth = 200;
  myIMU.settings.gyroBandWidth = 200;


  if (myIMU.begin() != 0) {
    return false;
  }

  BLE.setLocalName("DPOINT");
  BLE.setAdvertisedService(dpointService);
  BLE.setConnectionInterval(6, 6); // Units of 1.25ms

  imuCharacteristic.addDescriptor(imuDescriptor);

  dpointService.addCharacteristic(imuCharacteristic);
  BLE.addService(dpointService);

  IMUDataPacket initialPacket = { 0 };
  imuCharacteristic.writeValue(initialPacket);

  BLE.setPairable(false);

  BLE.advertise();
  return true;
}

float wakeUpThreshold = 500.0f; // degrees per second
float stayAwakeThreshold = 20.0f; // degrees per second
unsigned long stayAwakeTime = 1000*30; // milliseconds

void run_ble(BLEDevice central) {
  while (central.connected()) {
    unsigned long startTime = millis();
    IMUDataPacket packet;

    packet.pressure = pressureIn.read_u16();
    packet.accel[0] = myIMU.readRawAccelX();
    packet.accel[1] = myIMU.readRawAccelY();
    packet.accel[2] = myIMU.readRawAccelZ();
    packet.gyro[0] = myIMU.readRawGyroX();
    packet.gyro[1] = myIMU.readRawGyroY();
    packet.gyro[2] = myIMU.readRawGyroZ();
    
    imuCharacteristic.writeValue(packet);

    unsigned long time = millis() - startTime;
    unsigned long waitPeriod = delayMs - time;
    if (waitPeriod > 0 && waitPeriod < 500) { // protection against overflow issues
      delay(waitPeriod);
    }
  }
}

void sleep_stylus() {
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  // digitalWrite(PRESSURE_SENSOR_VCC_PIN, LOW);
  while (true) {
    float gX = myIMU.readFloatGyroX();
    if (abs(gX) > wakeUpThreshold) {
      // digitalWrite(PRESSURE_SENSOR_VCC_PIN, HIGH);
      return;
    }
    delay(100);
  }
}

void setup() {
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  // Set battery charge rate to 100mA
  pinMode(P0_13, OUTPUT);
  digitalWrite(P0_13, LOW);

  Wire1.setClock(400000UL);
  // Serial.begin(19200);
  setupPressureSensor();
}

unsigned long wakeUpTime;

void loop() {
  if (!start_ble()) {
    BLE.end();
    blink_error();
    return;
  }
  digitalWrite(LEDB, LOW);
  wakeUpTime = millis();
  while (millis() - wakeUpTime < stayAwakeTime) {
    BLEDevice central = BLE.central();
    if (central) {
      digitalWrite(LEDG, LOW);
      digitalWrite(LEDB, HIGH);
      run_ble(central);
      digitalWrite(LEDG, HIGH);
      digitalWrite(LEDB, LOW);
      BLE.end();
      digitalWrite(LEDB, HIGH);
      return;
    }
  }
  BLE.end();
  blink_error();
  sleep_stylus();
}
