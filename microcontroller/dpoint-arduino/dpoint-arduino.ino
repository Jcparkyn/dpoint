#include <LSM6DS3.h>
#include <ArduinoBLE.h>

const int ledPin = LED_BUILTIN; // set ledPin to on-board LED

const unsigned long rate = 120; // samples per second
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

void blink_error() {
  for (int i = 0; i < 11; i++) {
    digitalWrite(ledPin, (i % 2 == 0) ? HIGH : LOW);
    delay(200);
  }
}

bool start_ble() {
  digitalWrite(ledPin, LOW);
  if (!BLE.begin()) {
    return false;
  }

  myIMU.settings.accelRange = 4; // Can be: 2, 4, 8, 16
  myIMU.settings.gyroRange = 500; // Can be: 125, 245, 500, 1000, 2000
  myIMU.settings.accelSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666, 3332, 6664, 13330
  myIMU.settings.gyroSampleRate = 416; //Hz.  Can be: 13, 26, 52, 104, 208, 416, 833, 1666


  if (myIMU.begin() != 0) {
    return false;
  }

  BLE.setLocalName("DPOINT");
  BLE.setAdvertisedService(dpointService);
  BLE.setConnectionInterval(6, 8);

  imuCharacteristic.addDescriptor(imuDescriptor);

  dpointService.addCharacteristic(imuCharacteristic);
  BLE.addService(dpointService);

  IMUDataPacket initialPacket = { 0 };
  imuCharacteristic.writeValue(initialPacket);

  BLE.advertise();
  return true;
}

void stop_ble() {
  BLE.end();
  digitalWrite(ledPin, HIGH);
}

float wakeUpThreshold = 500.0f; // degrees per second
float stayAwakeThreshold = 20.0f; // degrees per second
unsigned long stayAwakeTime = 1000*60*2; // milliseconds

void run_ble() {
  unsigned long lastMotionTime = millis();
  while (true) {
    unsigned long startTime = millis();
    if (startTime - lastMotionTime > stayAwakeTime) {
      return;
    }

    BLE.poll();

    float aX = myIMU.readFloatAccelX();
    float aY = myIMU.readFloatAccelY();
    float aZ = myIMU.readFloatAccelZ();
    float gX = myIMU.readFloatGyroX();
    float gY = myIMU.readFloatGyroY();
    float gZ = myIMU.readFloatGyroZ();

    float motionAmount = gX*gX + gY*gY + gZ*gZ;
    if (motionAmount > stayAwakeThreshold*stayAwakeThreshold) {
      lastMotionTime = startTime;
    }
    
    IMUDataPacket packet = {
      .accel = {aX, aY, aZ},
      .gyro = {gX, gY, gZ},
      .time = startTime,
    };
    imuCharacteristic.writeValue(packet);

    unsigned long time = millis() - startTime;
    unsigned long waitPeriod = delayMs - time;
    if (waitPeriod > 0 && waitPeriod < 500) { // protection against overflow issues
      delay(waitPeriod);
    }
  }
}

void sleep() {
  digitalWrite(ledPin, HIGH);
  while (true) {
    float gX = myIMU.readFloatGyroX();
    float gY = myIMU.readFloatGyroY();
    float gZ = myIMU.readFloatGyroZ();

    float motionAmount = gX*gX + gY*gY + gZ*gZ;

    if (motionAmount > wakeUpThreshold*wakeUpThreshold) {
      return;
    }
    delay(200);
  }
}

void setup() {
  pinMode(ledPin, OUTPUT);
  blink_error();
}

void loop() {
  if (!start_ble()) {
    blink_error();
    return;
  }
  run_ble();
  stop_ble();
  blink_error();
  sleep();
}
