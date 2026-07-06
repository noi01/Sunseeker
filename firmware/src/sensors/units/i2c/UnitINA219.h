#pragma once
#ifndef MBK_INA219_H
#define MBK_INA219_H

#include <Arduino.h>
#include <ArduinoLog.h>
#include <Wire.h>
#include "INA219.h"
#include "sensors/units/i2c/UnitI2C.h"

//  INA219 bi-directional current / power monitor (I2C, default address 0x40).

class UnitINA219 : public UnitI2C
{
    //  Hardware
TwoWire  *_wire  = &Wire;
    static constexpr uint8_t k_address = 0x40;
    INA219 _sensor{0x40};
 
    //  Cached measurements

    float    _current = 0.0f;
    float   _voltage = 0.0f;
    float _shunt_voltage = 0.0f;
    float _power = 0;
public:

    UnitINA219()           : UnitI2C() {}
    UnitINA219(TwoWire& w) : UnitI2C(),  _wire(&w) {}

    //  Lifecycle

    bool begin_impl() override
    {
        Log.traceln("UnitINA219 begin_impl");
        _wire->begin();
        _connected = _sensor.begin();
          //  INA.setMaxCurrentShunt(1, 0.002);
        //  delay(1000);
        //  INA.setMaxCurrentShunt(2.5, 0.002);
        //  delay(1000);
        _sensor.setMaxCurrentShunt(5, 0.002);
        delay(1000);
        //  INA.setMaxCurrentShunt(7.5, 0.002);
        //  delay(1000);
        //  INA.setMaxCurrentShunt(10, 0.002);
        //  delay(1000);
        //  INA.setMaxCurrentShunt(15, 0.002);
        //  delay(1000);
        //  INA.setMaxCurrentShunt(20, 0.002);
        //  delay(10000);

        Serial.println(_sensor.getBusVoltageRange());
        _retry_delay_ms = k_retry_min_ms;
        _next_probe_ms = 0;
        return _connected;
    }

    bool sample_impl(uint32_t now_ms) override
    {
        if (!_connected)
        {
            if (!try_reconnect(now_ms))
            {
                return false;
            }
        }

            

        _current = _sensor.getCurrent_mA();
        _voltage = _sensor.getBusVoltage();
        _shunt_voltage =_sensor.getShuntVoltage_mV();
        _power = _sensor.getPower_mW();
  
        return true;
    }

    void teardown_impl() override {}

    //  Serialisation

    void write_json(JsonObject& dst) const override
    {   
        Unit::write_json(dst);
        dst["val"].add(_current);
        dst["val"].add(_voltage);
        dst["val"].add(_shunt_voltage);     
        dst["val"].add(_power);    
    }

    static void prefix_print(Print* _logOutput, int logLevel){
      _logOutput->printf("[Unit - INA219] ");
    }
};

#endif  // MBK__sensor_H