/**
 * @file OutputUnit.h
 * @author Etienne Montenegro
 * @brief Digital output unit driving a PortC analog signal line.
 *
 * OutputUnit treats the PortC analog line (pins[0]) as a plain digital
 * output. The state is set explicitly through set_output_value() (typically
 * from a websocket command) and held until changed. The line is never read,
 * so a port configured with this model cannot sample its analog input.
 */

#ifndef OUTPUT_UNIT_H
#define OUTPUT_UNIT_H


#include "sensors/units/Unit.h"
#include "sensors/ports/PortC.h"

class OutputUnit : public Unit {
 public:
  /**
   * @brief Construct an OutputUnit bound to a PortC GPIO resource.
   *
   * @param port  PortC that provides the digital output pin.
   *              The referenced PortC must outlive this object.
   * @param _model Unit model (expected sensor_output).
   */
  OutputUnit(PortC& port, UnitModel _model);

  /**
   * @brief Drive the output line to the requested state.
   *
   * @param value true = HIGH, false = LOW.
   * @return true if the port pin was driven, false if the unit is not
   *         initialized yet.
   */
  bool set_output_value(bool value) override;

  /**
   * @brief Serialize the current output state into a JSON object.
   *
   * Calls Unit::write_json first (id, name), then appends:
   *   "val" : bool   — current driven level
   */
  void write_json(JsonObject& dst) const override;

 protected:
  bool begin_impl() override;
  bool sample_impl(uint32_t now_ms) override;
  void teardown_impl() override;

 private:
  PortC& _port;
  bool _state{false};

  static void prefix_print(Print* _logOutput, int logLevel) {
    _logOutput->printf("[Unit - Output] ");
  }
};

#endif  // OUTPUT_UNIT_H
