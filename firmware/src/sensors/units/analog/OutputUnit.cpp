/**
 * @file OutputUnit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of OutputUnit.h
 */

#include "sensors/units/analog/OutputUnit.h"
#include <ArduinoLog.h>

OutputUnit::OutputUnit(PortC& port, UnitModel _model) : _port(port), Unit(_model) {
  set_id(_port.id);
}

bool OutputUnit::begin_impl() {
  _logger.setPrefix(prefix_print);
  pinMode(_port.pins[0], OUTPUT);
  digitalWrite(_port.pins[0], _state ? HIGH : LOW);
  return true;
}

bool OutputUnit::set_output_value(bool value) {
  _state = value;
  digitalWrite(_port.pins[0], _state ? HIGH : LOW);
  return true;
}

bool OutputUnit::sample_impl(uint32_t now_ms) {
  (void)now_ms;
  return true;
}

void OutputUnit::write_json(JsonObject& dst) const {
  Unit::write_json(dst);
  dst["val"] = _state;
}

void OutputUnit::teardown_impl() {
}
