#include <TinyGPS++.h>

TinyGPSPlus gps;

void sendUBX(uint8_t *msg, uint8_t len) {
  for (uint8_t i = 0; i < len; i++) {
    Serial1.write(msg[i]);
  }
}

void setup() {

  Serial.begin(9600);
  Serial1.begin(9600);

  Serial.println("GPS start");

  // Set update rate to 5Hz
  uint8_t rate[] = {
    0xB5,0x62,0x06,0x08,
    0x06,0x00,
    0xC8,0x00,   // 200ms = 5Hz
    0x01,0x00,
    0x01,0x00,
    0xDE,0x6A
  };

  sendUBX(rate, sizeof(rate));
}

void loop() {

  while (Serial1.available()) {
    gps.encode(Serial1.read());
  }

  if (gps.location.isUpdated()) {

    Serial.print("Satellites: ");
    Serial.println(gps.satellites.value());

    Serial.print("Latitude: ");
    Serial.println(gps.location.lat(), 6);

    Serial.print("Longitude: ");
    Serial.println(gps.location.lng(), 6);

    Serial.print("Altitude: ");
    Serial.println(gps.altitude.meters());

    Serial.print("HDOP: ");
    Serial.println(gps.hdop.value() / 100.0);

    Serial.println();
  }
}
