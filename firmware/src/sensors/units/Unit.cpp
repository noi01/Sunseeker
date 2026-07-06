/**
 * @file Unit.cpp
 * @author Etienne Montenegro
 * @brief Implementation of the Unit lifecycle base class.
 */

#include <Arduino.h>
#include <ArduinoLog.h>
#include "sensors/units/Unit.h"

Unit::Unit(UnitModel _model): model(_model){

}

bool Unit::begin() {
    _logger.begin(LOG_LEVEL_TRACE,&Serial);
    _initialized = begin_impl();

  if (!_initialized) {
    _logger.errorln("Failed to initialize Unit model=%s on port %d", model_to_str(model), _id);
    return false;
  }
  _logger.infoln("Success connecting to device");
  return true;
}
 void Unit::write_json(JsonObject& dst) const{
    dst["id"] = _id;
    dst["name"] = model_to_str(model);
  };

void Unit::teardown() {
  _logger.traceln("Unit::teardown() id=%d model=%s", _id, model_to_str(model));
  teardown_impl();
  _enable = false;
}

bool Unit::tick(uint32_t now_ms) {
  
  if (!_enable) {
      // _logger.traceln("id=%d model=%s Not enabled", _id, model_to_str(model));
    return false;
  }

  if ((now_ms - _last_sample_ms) < _sample_period_ms) {
    return false;
  }

  const bool ok = sample_impl(now_ms);
  _last_sample_ms = now_ms;

  if (ok) {
    return true;
  }

  return false;
}


void Unit::set_enabled(bool val) {
  if(! _initialized){
    _logger.warningln("Unit %d not initialized, cannot enable",_id);
    _enable = false;
    return;
  }
  _logger.traceln("Unit::set_enabled() id=%d enabled=%d", _id, val);
  _enable = val;

}

float Unit::map_to_float(float val, float min_range, float max_range, float new_min, float new_max){
  return (val - min_range) * ((new_max - new_min) / (max_range - min_range)) + new_min;
}

float Unit::apply_lowpass(float current, float previous){
  float a = get_alpha();
  return (a * current) + ((1.0f - a) * previous);
}

