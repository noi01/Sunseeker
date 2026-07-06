// /**
//  * @file main.cpp
//  * @authors Etienne Montenegro (MxLab - UQAM)
//  *          Jerome Saint-Clair (Ensad)
//  * @brief Main file of the MisBKit project
//  *
//  *
//  */

#include <Arduino.h>
#include <ArduinoLog.h>
#ifdef WEB_SERIAL
#include <WebSerial.h>
#define LOG_DESTINATION &WebSerial
#else
#define LOG_DESTINATION &Serial
#endif
#include <ServoEasing.hpp>
#include "MisBKit.h"

MisBKit mbk{};

void setup()
{
  Serial.begin(115200);
  // while (!Serial)
  // {
  // }
  delay(5000);
  Log.begin(LOG_LEVEL_TRACE, LOG_DESTINATION);
  Log.warningln("MisBKit %d.%d.%d", V_MAJ, V_MIN, V_PATCH);
  mbk.initialize(false);
  Log.warningln("End on setup.");
}

void loop()
{
  mbk.update();
}


// #include <Arduino.h>

// #include <Wire.h>

// // Set I2C bus to use: Wire, Wire1, etc.
// #define WIRE Wire

// void setup() {
//   WIRE.setPins(SCL, SDA);
//   WIRE.begin();

//   Serial.begin(115200);
  
//      delay(2000);
//   Serial.println("\nI2C Scanner");
// }


// void loop() {
//   byte error, address;
//   int nDevices;

//   Serial.println("Scanning...");

//   nDevices = 0;
//   for(address = 1; address < 127; address++ )
//   {
//     // The i2c_scanner uses the return value of
//     // the Write.endTransmisstion to see if
//     // a device did acknowledge to the address.
//     WIRE.beginTransmission(address);
//     error = WIRE.endTransmission();

//     if (error == 0)
//     {
//       Serial.print("I2C device found at address 0x");
//       if (address<16)
//         Serial.print("0");
//       Serial.print(address,HEX);
//       Serial.println("  !");

//       nDevices++;
//     }
//     else if (error==4)
//     {
//       Serial.print("Unknown error at address 0x");
//       if (address<16)
//         Serial.print("0");
//       Serial.println(address,HEX);
//     }
//   }
//   if (nDevices == 0)
//     Serial.println("No I2C devices found\n");
//   else
//     Serial.println("done\n");

//   delay(5000);           // wait 5 seconds for next scan
// }