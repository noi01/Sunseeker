/**
 * @file PortA.cpp
 * @author Etienne Montenegro
 * @brief 
 */

#include "PortA.h"

#include <ArduinoLog.h>
#include <ArduinoJson.h>
#include "sensors/units/Unit.h"
#include "sensors/units/i2c/I2CUnits.h"
#include "sensors/units/i2c/UnitPAHub.h"



PortA::~PortA() = default;


void PortA::prefix_print(Print* _logOutput, int logLevel) {
	_logOutput->print("[PortA] ");
}
bool PortA::begin() {
  _logger.begin(LOG_LEVEL_TRACE,&Serial);
  _logger.setPrefix(&PortA::prefix_print);
  if (_bus_initialized) {
    _logger.traceln("Port %d of type A already initialized", id);
    return true;
  }

  Wire.setPins(pins[1], pins[0]);

  if (!Wire.begin()) {
    _logger.errorln("Failed to initialize port %d of type A on pins scl: %d sda: %d",id, pins[0], pins[1]);
    return false;
  }
  Wire.setTimeOut(100);
  Wire.setClock(100000);
  _bus_initialized = true;
  return true;
}

bool PortA::begin_bus_if_needed() {
  if (_bus_initialized) {
    return true;
  }
  return begin();
}



DetectedModels PortA::autodetect_units() {
  _logger.traceln("Starting unit scan");
  
      //clean a potential blocked device:
    for (int i = 0; i < 9; i++) {
        digitalWrite(pins[1], HIGH);
        delayMicroseconds(5);
        digitalWrite(pins[1], LOW);
        delayMicroseconds(5);
    }
  
  DetectedModels detected;

  if (!begin_bus_if_needed()) {
    _logger.warningln("Port %d autodetect_units() bus init failed", id);
    return detected;
  }

  for (uint8_t address = 0x01; address < 0x7f; address++) {
    Wire.beginTransmission(address);
    uint8_t error = Wire.endTransmission();
    if (error != static_cast<uint8_t>(WireStatus::SUCCESS)) {
      continue;
    }
    UnitModel m = model_from_i2c_address(address);
    _logger.traceln("Port %d autodetect_units() found %s at %X  (%d)", id, model_to_str(m), address, address);
    if (m != UnitModel::none) {
      detected.push(m);
    }
  }

  // if using an i2c multiplexer, it is possibe that a unit connected on it was scanned.
  // to prevent this and make sure the unit is not duplicated, when scanning an i2c multiplexer
  // delete all found units and only assign the multiplexer to the port.
  // The units connected to it will live inside the pahub object
  bool overwrite = false;
  for(uint8_t i = 0 ; i < detected.count ; i ++){
    if(detected.models[i] == UnitModel::sensor_pahub){
      overwrite = true;
    }
  }

  if(overwrite){
    detected = DetectedModels{};
    detected.push( UnitModel::sensor_pahub);
  }
  _logger.traceln("Scan done");
  return detected;
}

int PortA::find_next_index(){
    uint8_t free_index = 0;
  while (free_index < k_max_units && _owned_units[free_index]) {
    free_index++;
  }
  return free_index;
}
Unit* PortA::create_unit(UnitModel model, int index, uint8_t address) {
  if (index < 0 || index >= k_max_units) {
    _logger.warningln("Port %d create_unit() invalid index=%d (max=%d)", id, index, k_max_units);
    return nullptr;
  }

  std::unique_ptr<Unit> unit;
  TwoWire& bus = *get_interface();
  switch (model) {
    case UnitModel::sensor_tof4m:
      unit = std::make_unique<UnitVL53L1X>(bus);
      break;
    case UnitModel::sensor_tvoc:
      unit = std::make_unique<UnitSGP30>(bus);
      break;
    case UnitModel::sensor_dlight:
      unit = std::make_unique<UnitDLight>(bus);
      break;
    case UnitModel::sensor_accel:
      unit = std::make_unique<MPU6886>(bus);
      break;
    case UnitModel::sensor_ultrasonic_i2c:
      unit = std::make_unique<UltrasonicI2C>(bus);
      break;
    case UnitModel::hmi_8angle:
      unit = std::make_unique<Unit8Angle>(bus);
      break;
    case UnitModel::sensor_pahub:
      if(address >= 70 && address <=77){
        unit = std::make_unique<UnitPAHUB>(bus, address);  
      }else{
        unit = std::make_unique<UnitPAHUB>(bus);
      }
      break;
    case UnitModel::sensor_ina219:
      unit = std::make_unique<UnitINA219>(bus);
     break;
    case UnitModel::sensor_max17048:
      unit = std::make_unique<UnitMAX17048>(bus);
     break;
    default:
      _logger.warningln("Port %d create_unit() unsupported model %d", id, static_cast<int>(model));
      return nullptr;
  }

  if (!unit) {
    return nullptr;
  }
  _logger.traceln("id:%d Created unit[%d] of model=%s", id, index, model_to_str(model));


  unit->set_model(model);
  unit->set_id(id);
  if (!_owned_units[index]) {
    _owned_unit_count++;
  }
  _owned_units[index] = std::move(unit);
  return _owned_units[index].get();
}

bool PortA::adopt_unit(std::unique_ptr<Unit> unit) {
  if (!unit) {
    return false;
  }

  if (!begin_bus_if_needed()) {
    return false;
  }

  uint8_t free_index = 0;
  while (free_index < k_max_units && _owned_units[free_index]) {
    free_index++;
  }

  if (free_index >= k_max_units) {
    return false;
  }

  unit->set_id(id);
  _owned_units[free_index] = std::move(unit);
  _owned_unit_count++;
  return true;
}

void PortA::remove_unit(const Unit* unit) {
  if (!unit) {
    return;
  }

  for (uint8_t i = 0; i < k_max_units; ++i) {
    if (_owned_units[i] && _owned_units[i].get() == unit) {
      _owned_units[i].reset();
      if (_owned_unit_count > 0) {
        --_owned_unit_count;
      }
      return;
    }
  }
}

void PortA::clear_units() {
  for (auto &owned : _owned_units) {
    owned.reset();
  }
  _owned_unit_count = 0;
}

uint8_t PortA::unit_count() const {
  return _owned_unit_count;
}

Unit* PortA::unit_at(uint8_t index) const {
  if (index >= k_max_units) {
    return nullptr;
  }
  return _owned_units[index].get();
}

void PortA::tick_units(uint32_t now) {
  for (const auto& owned : _owned_units) {
    if (owned) {
      owned->tick(now);
    }
  }
}

void PortA::serialize_units(JsonArray& arr) {
  for (const auto& owned : _owned_units) {
    if (owned && owned->is_enabled()) {
      JsonObject obj = arr.add<JsonObject>();
      owned->write_json(obj);
    }
  }
}

bool PortA::has_model(UnitModel model) const {
  for (const auto& owned : _owned_units) {
    if (owned && owned->get_model() == model) {
      return true;
    }
  }
  return false;
}

bool PortA::attach_unit(Unit* unit) {
  _logger.traceln("Port %d attach_unit() entry unit=%p unit_count=%d", id, unit, _attached_unit_count);

  if (!unit) {
    _logger.warningln("Port %d attach_unit() null unit pointer - aborting", id);
    return false;
  }

  if (!begin_bus_if_needed()) {
    _logger.warningln("Port %d attach_unit() failed to initialize bus", id);
    return false;
  }

  for (uint8_t i = 0; i < _attached_unit_count; i++) {
    if (_attached_units[i] == unit) {
      return true;
    }
  }

  if (_attached_unit_count >= k_max_units) {
    _logger.warningln("Port %d attach_unit() cannot attach unit=%p - capacity reached (%d/%d)", id, unit, _attached_unit_count, k_max_units);
    return false;
  }

  _attached_units[_attached_unit_count++] = unit;
  _logger.infoln("Port %d attach_unit() attached unit=%p new_unit_count=%d", id, unit, _attached_unit_count);
  return true;
}

void PortA::detach_unit(const Unit* unit) {
  if (!unit) {
    return;
  }

  for (uint8_t i = 0; i < _attached_unit_count; i++) {
    if (_attached_units[i] != unit) {
      continue;
    }

    for (uint8_t j = i; j + 1 < _attached_unit_count; j++) {
      _attached_units[j] = _attached_units[j + 1];
    }
    _attached_units[_attached_unit_count - 1] = nullptr;
    _attached_unit_count--;
    return;
  }
}

Unit* PortA::connected_unit(uint8_t index) const {
  if (index >= _attached_unit_count) {
    return nullptr;
  }
  return _attached_units[index];
}


