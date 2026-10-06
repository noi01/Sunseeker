/**
 * @file AnalogInputUnit.h
 * @author Etienne Montenegro
 */

#ifndef ANALOG_INPUT_UNIT_H
#define ANALOG_INPUT_UNIT_H


#include "sensors/units/Unit.h"
#include "sensors/ports/PortC.h"

class AnalogInputUnit : public Unit {
 public:
  AnalogInputUnit(PortC& port, UnitModel _model);

  void write_json(JsonObject& dst) const override;

 protected:
  bool begin_impl() override;
  bool sample_impl(uint32_t now_ms) override;
  void teardown_impl() override;

 private:
  PortC& _port;
  static void prefix_print(Print* _logOutput, int logLevel) {
    _logOutput->printf("[Unit - AnalogInput] ");
  }
};

#endif  // ANALOG_INPUT_UNIT_H
