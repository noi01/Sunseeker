#include "sensors/ports/PortC.h"

#include "sensors/units/analog/AnalogUnits.h"


PortC::~PortC() = default;

bool PortC::begin() {
  return true;
}

Unit* PortC::create_unit(UnitModel model, int index, uint8_t address) {
  if (index < 0 || index >= k_max_units) {
    _logger.warningln("Port %d create_unit() invalid index=%d (max=%d)", id, index, k_max_units);
    return nullptr;
  }
  std::unique_ptr<Unit> unit;
  switch (model) {
    case UnitModel::sensor_generic:
    case UnitModel::sensor_light:
    case UnitModel::sensor_mic:
      unit.reset(new GenericUnit(*this, model));
      break;
    case UnitModel::hmi_fader:
      unit.reset(new FaderUnit(*this, model));
      break;
    case UnitModel::hmi_button:
      unit.reset(new ButtonUnit(*this, model));
      break;
    default:
      return nullptr;
  }

  if (!unit) {
    return nullptr;
  }
   _logger.warningln("Port %d create_unit() unsupported model %d", id, static_cast<int>(model));

  unit->set_model(model);
  unit->set_id(id);
  _owned_unit = std::move(unit);
  return _owned_unit.get();
}

bool PortC::adopt_unit(std::unique_ptr<Unit> unit) {
  if (!unit || _owned_unit) {
    return false;
  }

  unit->set_id(id);
  _owned_unit = std::move(unit);
  return true;
}

void PortC::remove_unit(const Unit* unit) {
  if (!unit || !_owned_unit) {
    return;
  }

  if (_owned_unit.get() == unit) {
    _owned_unit.reset();
  }
}

void PortC::clear_units() {
  _owned_unit.reset();
}

uint8_t PortC::unit_count() const {
  return _owned_unit ? 1 : 0;
}

Unit* PortC::unit_at(uint8_t index) const {
  if (index != 0 || !_owned_unit) {
    return nullptr;
  }
  return _owned_unit.get();
}

void PortC::tick_units(uint32_t now) {
  if (_owned_unit) {
    _owned_unit->tick(now);
  }
}

void PortC::serialize_units(JsonArray& arr) {
  if (_owned_unit && _owned_unit->is_enabled()) {
    JsonObject obj = arr.add<JsonObject>();
    _owned_unit->write_json(obj);
  }
}

bool PortC::has_model(UnitModel model) const {
  return _owned_unit && _owned_unit->get_model() == model;
}
