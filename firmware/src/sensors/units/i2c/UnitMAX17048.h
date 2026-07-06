#pragma once
#ifndef MBK_MAX17048_H
#define MBK_MAX17048_H

#include <Arduino.h>
#include <ArduinoLog.h>
#include <Wire.h>
#include <Adafruit_MAX1704X.h>
#include "sensors/units/i2c/UnitI2C.h"

//  MAX17048 1-cell Li-Ion / LiPo fuel-gauge (I2C, fixed address 0x36).

class UnitMAX17048 : public UnitI2C
{
  //  Hardware
  TwoWire  *_wire  = &Wire;
  Adafruit_MAX17048   _gauge;

  //  Cached measurements

  float _cellVoltage = 0.0f;
  float _percent = 0.0f;
  float _rate = 0.0f;

public:

  UnitMAX17048()           : UnitI2C() {}
  UnitMAX17048(TwoWire& w) : UnitI2C(), _wire(&w) {}

  //  Lifecycle

  bool begin_impl() override
  {
    Log.traceln("UnitMAX17048 begin_impl wire=%p", _wire);
    if(!_gauge.begin(_wire)){
      _connected= false;
      return false;
    }
    _connected = true;  // non-zero silicon ID confirms device is reachable
    _retry_delay_ms = k_retry_min_ms;
    _next_probe_ms = 0;
    Serial.print(F("Found MAX17048"));
  Serial.print(F(" with Chip ID: 0x")); 
  Serial.println(_gauge.getChipID(), HEX);

  // The reset voltage is what the chip considers 'battery has been removed and replaced'
  // The default is 3.0 Volts but you can change it here: 
  //_gauge.setResetVoltage(2.5);
  Serial.print(F("Reset voltage = ")); 
  Serial.print(_gauge.getResetVoltage());
  Serial.println(" V");

  // Hibernation mode reduces how often the ADC is read, for power reduction. There is an automatic
  // enter/exit mode but you can also customize the activity threshold both as voltage and charge rate

  //_gauge.setActivityThreshold(0.15);
  Serial.print(F("Activity threshold = ")); 
  Serial.print(_gauge.getActivityThreshold()); 
  Serial.println(" V change");

  //_gauge.setHibernationThreshold(5);
  Serial.print(F("Hibernation threshold = "));
  Serial.print(_gauge.getHibernationThreshold()); 
  Serial.println(" %/hour");

  // The alert pin can be used to detect when the voltage of the battery goes below or
  // above a voltage, you can also query the alert in the loop.
  _gauge.setAlertVoltages(2.0, 4.2);


    return _connected;
  }

  bool sample_impl(uint32_t now_ms) override
  {
    if (!_connected) {
      if (!try_reconnect(now_ms)) {
        return false;
      }
    }

  _cellVoltage = _gauge.cellVoltage();
  _percent = _gauge.cellPercent();
  _rate = _gauge.chargeRate();
  if (isnan(_cellVoltage)) {
    Serial.println("Failed to read cell voltage, check battery is connected!");
    delay(2000);
    return false;
  }






  if (_gauge.isActiveAlert()) {
    uint8_t status_flags = _gauge.getAlertStatus();
    Serial.print(F("ALERT! flags = 0x"));
    Serial.print(status_flags, HEX);
    
    if (status_flags & MAX1704X_ALERTFLAG_SOC_CHANGE) {
      Serial.print(", SOC Change");
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_SOC_CHANGE); // clear the alert
    }
    if (status_flags & MAX1704X_ALERTFLAG_SOC_LOW) {
      Serial.print(", SOC Low");
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_SOC_LOW); // clear the alert
    }
    if (status_flags & MAX1704X_ALERTFLAG_VOLTAGE_RESET) {
      Serial.print(", Voltage reset");
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_VOLTAGE_RESET); // clear the alert
    }
    if (status_flags & MAX1704X_ALERTFLAG_VOLTAGE_LOW) {
      Serial.print(", Voltage low ");
      Serial.print(_cellVoltage);
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_VOLTAGE_LOW); // clear the alert
    }
    if (status_flags & MAX1704X_ALERTFLAG_VOLTAGE_HIGH) {
      Serial.print(", Voltage high");
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_VOLTAGE_HIGH); // clear the alert
    }
    if (status_flags & MAX1704X_ALERTFLAG_RESET_INDICATOR) {
      Serial.print(", Reset Indicator");
      _gauge.clearAlertFlag(MAX1704X_ALERTFLAG_RESET_INDICATOR); // clear the alert
    }
    Serial.println();
    
  }
    return true;
  }

  void teardown_impl() override {}

  //  Serialisation

  void write_json(JsonObject& dst) const override
  {
        Unit::write_json(dst);
        dst["val"].add(_cellVoltage);
        dst["val"].add(_percent);
        dst["val"].add(_rate); 
  }
  static void prefix_print(Print* _logOutput, int logLevel){
      _logOutput->printf("[Unit - MAX17048] ");
    }
};

#endif  // MBK_MAX17048_H
