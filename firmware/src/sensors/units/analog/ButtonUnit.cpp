/**
 * @file ButtonUnit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of ButtonUnit.h
 */

#include "sensors/units/analog/ButtonUnit.h"
#include <ArduinoLog.h>

ButtonUnit::ButtonUnit(PortC& port, UnitModel _model) : _port(port), Unit(_model){
    set_id(_port.id);
}


bool ButtonUnit::begin_impl() {
    _logger.setPrefix(prefix_print);
    pinMode(_port.pins[0],INPUT);
    return true;
}

bool ButtonUnit::sample_impl(uint32_t now_ms) {

    raw[0] = analogRead(_port.pins[0]);
    
    _filtered[0] = apply_lowpass(map_to_float(raw[0],0,4095), _filtered[0]);
    
    return true;
}


void ButtonUnit::write_json(JsonObject& dst) const {
    Unit::write_json(dst);
    dst["val"].add(_filtered[0]);
}

void ButtonUnit::teardown_impl(){
}


