#include <LSM6DS3.h>
#include <ArduinoBLE.h>

const int ledPin = LED_BUILTIN; // set ledPin to on-board LED

const unsigned long rate = 100; // samples per second
const unsigned long delayMs = 1000/rate;

struct IMUDataPacket {
  float accel[3];
  float gyro[3];
  uint32_t time;
};

BLEService dpointService("19B10010-E8F2-537E-4F6C-D104768A1214");

BLEDescriptor imuDescriptor("2901", "IMU");
BLETypedCharacteristic<IMUDataPacket> imuCharacteristic("19B10013-E8F2-537E-4F6C-D104768A1214", BLERead | BLENotify);

LSM6DS3 myIMU(I2C_MODE, 0x6A); //I2C device address 0x6A

void setup() {
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, LOW);

  if (!BLE.begin()) {
    while (1);
  }

  if (myIMU.begin() != 0) {
    Serial.println("Device error");
    while (1);
  } else {
    Serial.println("aX,aY,aZ,gX,gY,gZ");
  }

  BLE.setLocalName("DPOINT");
  BLE.setAdvertisedService(dpointService);
  BLE.setConnectionInterval(6, 8);

  imuCharacteristic.addDescriptor(imuDescriptor);

  dpointService.addCharacteristic(imuCharacteristic);
  BLE.addService(dpointService);

  IMUDataPacket initialPacket = { 0 };
  imuCharacteristic.writeValue(initialPacket);

  // start advertising
  BLE.advertise();
}

void loop() {
  unsigned long startTime = millis();

  BLE.poll();

  float aX = myIMU.readFloatAccelX();
  float aY = myIMU.readFloatAccelY();
  float aZ = myIMU.readFloatAccelZ();
  float gX = myIMU.readFloatGyroX();
  float gY = myIMU.readFloatGyroY();
  float gZ = myIMU.readFloatGyroZ();
  
  IMUDataPacket packet = {
    .accel = {aX, aY, aZ},
    .gyro = {gX, gY, gZ},
    .time = millis(),
  };
  imuCharacteristic.writeValue(packet);

  unsigned long time = millis() - startTime;
  unsigned long waitPeriod = delayMs - time;
  if (waitPeriod > 0 && waitPeriod < 500) { // protection against overflow issues
    delay(waitPeriod);
  }
}
