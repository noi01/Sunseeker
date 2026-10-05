#pragma once
#ifndef MBK_WATER_LEVEL_H
#define MBK_WATER_LEVEL_H

#include <Arduino.h>
#include <ArduinoLog.h>
#include <Wire.h>
#include "waterlevelsensor.h"
#include "sensors/units/i2c/UnitI2C.h"

//  Grove Water Level Sensor (I2C). Two on-board ATTiny co-processors expose
//  the strip as capacitive sections:
//      0x77 - low  8 sections
//      0x78 - high 12 sections
//  The library returns the fill level as a percentage (0..100).

class UnitWaterLevel : public UnitI2C
{
    //  Hardware
    TwoWire *_wire = &Wire;
    static constexpr uint8_t k_low_address  = 0x77;
    static constexpr uint8_t k_high_address = 0x78;
    WaterLevelSensor _sensor;

    //  Cached measurements
    int   _percentage = 0;
    float _filtered_percentage = 0.0f;

    bool is_present(uint8_t address)
    {
        _wire->beginTransmission(address);
        return _wire->endTransmission() == 0;
    }

public:

    UnitWaterLevel()             : UnitI2C() {}
    UnitWaterLevel(TwoWire& w)   : UnitI2C(), _wire(&w) {}

    //  Lifecycle

    bool begin_impl() override
    {
        Log.traceln("UnitWaterLevel begin_impl wire=%p", _wire);
        _wire->begin();

        //  The high module (0x78) is unique to this sensor and is the
        //  canonical probe address; the low module (0x77) overlaps the
        //  PAHUB address range and is therefore not used for detection.
        _connected = is_present(k_high_address) && is_present(k_low_address);
        if (!_connected) {
            _logger.errorln("Water level sensor not found (0x%02X/0x%02X)", k_high_address, k_low_address);
            return false;
        }

        _percentage = 0;
        _filtered_percentage = 0.0f;
        _retry_delay_ms = k_retry_min_ms;
        _next_probe_ms = 0;
        return true;
    }

    bool sample_impl(uint32_t now_ms) override
    {
        if (!_connected) {
            if (!try_reconnect(now_ms)) {
                return false;
            }
        }

        //  readPercentage() blocks forever waiting for the two modules, so
        //  verify the hardware is still responding before calling into it.
        if (!is_present(k_high_address) || !is_present(k_low_address)) {
            note_disconnect(now_ms);
            return false;
        }

        _percentage = _sensor.readPercentage();
        _filtered_percentage = apply_lowpass(static_cast<float>(_percentage), _filtered_percentage);

        return true;
    }

    void teardown_impl() override {}

    //  Serialisation

    void write_json(JsonObject& dst) const override
    {
        Unit::write_json(dst);
        dst["val"].add(_percentage);
        dst["val"].add(_filtered_percentage);
    }

    static void prefix_print(Print* _logOutput, int logLevel){
      _logOutput->printf("[Unit - WaterLevel] ");
    }
};

#endif  // MBK_WATER_LEVEL_H
