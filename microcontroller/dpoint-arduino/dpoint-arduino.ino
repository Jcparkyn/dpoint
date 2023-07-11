#include <LSM6DS3.h>
#include <ArduinoBLE.h>

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

    float aX = myIMU.readFloatAccelX();
    float aY = myIMU.readFloatAccelY();
    float aZ = myIMU.readFloatAccelZ();
    float gX = myIMU.readFloatGyroX();
    float gY = myIMU.readFloatGyroY();
    float gZ = myIMU.readFloatGyroZ();
    
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
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  while (true) {
    float gX = myIMU.readFloatGyroX();
    if (abs(gX) > wakeUpThreshold) {
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
  sleep();
}
