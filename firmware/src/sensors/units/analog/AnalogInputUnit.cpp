/**
 * @file AnalogInputUnit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of AnalogInputUnit.h
 */

#include "sensors/units/analog/AnalogInputUnit.h"
#include <ArduinoLog.h>

AnalogInputUnit::AnalogInputUnit(PortC& port, UnitModel _model) : _port(port), Unit(_model) {
  set_id(_port.id);
}

bool AnalogInputUnit::begin_impl() {
  _logger.setPrefix(prefix_print);
  pinMode(_port.pins[0], INPUT);
  return true;
}

bool AnalogInputUnit::sample_impl(uint32_t now_ms) {
  (void)now_ms;

  raw[0] = analogRead(_port.pins[0]);

  _filtered[0] = apply_lowpass(map_to_float(raw[0], 0, 4095), _filtered[0]);

  return true;
}

void AnalogInputUnit::write_json(JsonObject& dst) const {
  Unit::write_json(dst);
  dst["val"].add(_filtered[0]);
}

void AnalogInputUnit::teardown_impl() {
}
