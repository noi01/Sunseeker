/**
 * @file GenericUnit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of GenericUnit.h
 */

#include "sensors/units/analog/GenericUnit.h"
#include <ArduinoLog.h>

GenericUnit::GenericUnit(PortC& port, UnitModel _model) : _port(port), Unit(_model){
    set_id(_port.id);
}


bool GenericUnit::begin_impl() {
    _logger.setPrefix(prefix_print);
    pinMode(_port.pins[0],INPUT);
    pinMode(_port.pins[1],INPUT);
    return true;
}

bool GenericUnit::sample_impl(uint32_t now_ms) {

    raw[0] = analogRead(_port.pins[0]);
    raw[1] = digitalRead(_port.pins[1]);
    
    _filtered[0] = apply_lowpass(map_to_float(raw[0],0,4095), _filtered[0]);
    _filtered[1] = apply_lowpass(map_to_float(raw[1],0,1), _filtered[1]);
    
    return true;
}


void GenericUnit::write_json(JsonObject& dst) const {
    Unit::write_json(dst);
    dst["val"].add(_filtered[0]);
    dst["val"].add(_filtered[1]);
}

void GenericUnit::teardown_impl(){
}


