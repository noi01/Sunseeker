#include "sensors/ports/Port.h"
#include "sensors/units/Unit.h"

Port::~Port() = default;

bool Port::adopt_unit(std::unique_ptr<Unit> unit) {
  (void)unit;
  return false;
}

void Port::set_enabled(bool val) {
  _enable = val;
}

DetectedModels Port::autodetect_units() {
  // Default no-op scan for ports that do not implement scanning.
}
