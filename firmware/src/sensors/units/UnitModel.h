/**
 * @file UnitModel.h
 * @author Etienne Montenegro
 * @brief Concrete module model selected by the user for a port.
 */

#ifndef UNIT_MODEL_H
#define UNIT_MODEL_H

#include <cstring>
#include <stdint.h>



enum class UnitModel : uint8_t {
  none = 0,

  // Generic profiles
  sensor_generic,
  
  // Sensor modules
  sensor_mic,
  sensor_light,
  sensor_ultrasonic_io,
  sensor_tof4m,
  sensor_tvoc,
  sensor_ultrasonic_i2c,
  sensor_dlight,
  sensor_accel,
  sensor_pahub,
  sensor_max17048,
  sensor_ina219,
  
  // HMI modules
  hmi_button,
  hmi_fader,
  hmi_angle,
  hmi_8angle
};

// i2c_address == 0 means the model is not I2C-detectable.
struct UnitModelEntry {
  UnitModel   model;
  const char* name;
  uint8_t     i2c_address;
};

struct UnitSpec {
    UnitModel model{UnitModel::none};
    bool enabled{false};
    float alpha{0.5f};
    int addr{-1};
};


// Keep this table in sync with the UnitModel enum.
constexpr UnitModelEntry k_unit_model_map[] = {
    {UnitModel::none,                  "none",                  0},
    {UnitModel::sensor_generic,        "sensor_generic",        0},
    {UnitModel::sensor_mic,            "sensor_mic",            0},
    {UnitModel::sensor_light,          "sensor_light",          0},
    {UnitModel::sensor_ultrasonic_io,  "sensor_ultrasonic_io",  0},
    {UnitModel::sensor_tof4m,          "sensor_tof4m",          0x29},
    {UnitModel::sensor_tvoc,           "sensor_tvoc",           0x58},
    {UnitModel::sensor_ultrasonic_i2c, "sensor_ultrasonic_i2c", 0x57},
    {UnitModel::sensor_dlight,         "sensor_dlight",         0x23},
    {UnitModel::sensor_accel,          "sensor_accel",          0x68},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x70},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x71},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x72},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x73},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x74},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x75},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x76},
    {UnitModel::sensor_pahub,          "sensor_pahub",          0x77},
    {UnitModel::sensor_ina219,         "sensor_ina219",         0x40},
    {UnitModel::sensor_max17048,       "sensor_max17048",       0x36},
    {UnitModel::hmi_button,            "hmi_button",            0},
    {UnitModel::hmi_fader,             "hmi_fader",             0},
    {UnitModel::hmi_angle,             "hmi_angle",             0},
    {UnitModel::hmi_8angle,            "hmi_8angle",            0x43},
};

inline const char* model_to_str(UnitModel m) {
  for (const auto& entry : k_unit_model_map) {
    if (entry.model == m) {
      return entry.name;
    }
  }

  return "none";
}

inline UnitModel model_str_to_enum(const char* model) {
  if (model == nullptr) {
    return UnitModel::none;
  }

  for (const auto& entry : k_unit_model_map) {
    if (strcmp(entry.name, model) == 0) {
      return entry.model;
    }
  }

  return UnitModel::none;
}

inline UnitModel model_from_i2c_address(uint8_t address) {
  for (const auto& entry : k_unit_model_map) {
    if (entry.i2c_address != 0 && entry.i2c_address == address) {
      return entry.model;
    }
  }
  return UnitModel::none;
}

constexpr uint8_t count_i2c_entries() {
  uint8_t n = 0;
  for (const auto& entry : k_unit_model_map) {
    if (entry.i2c_address != 0) n++;
  }
  return n;
}

// Fixed-size result type for I2C autodetection — no heap allocation.
constexpr uint8_t k_i2c_detectable_count = count_i2c_entries();

struct DetectedModels {
  UnitModel models[k_i2c_detectable_count]{};
  uint8_t   count{0};

  void push(UnitModel m) {
    if (count < k_i2c_detectable_count) {
      models[count++] = m;
    }
  }
};


#endif  // UNIT_MODEL_H