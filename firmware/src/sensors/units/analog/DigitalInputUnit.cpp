/**
 * @file DigitalInputUnit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of DigitalInputUnit.h
 */

#include "sensors/units/analog/DigitalInputUnit.h"
#include <ArduinoLog.h>

DigitalInputUnit::DigitalInputUnit(PortC& port, UnitModel _model) : _port(port), Unit(_model) {
  set_id(_port.id);
}

bool DigitalInputUnit::begin_impl() {
  _logger.setPrefix(prefix_print);
  pinMode(_port.pins[1], INPUT);
  return true;
}

bool DigitalInputUnit::sample_impl(uint32_t now_ms) {
  (void)now_ms;

  raw[0] = digitalRead(_port.pins[1]);
  _filtered[0] = static_cast<float>(raw[0]);

  return true;
}

void DigitalInputUnit::write_json(JsonObject& dst) const {
  Unit::write_json(dst);
  dst["val"].add(_filtered[0]);
}

void DigitalInputUnit::teardown_impl() {
}
