#ifndef MBK_CONFIGURATION_H
#define MBK_CONFIGURATION_H

#include <Arduino.h>
#include <ArduinoLog.h>
#include <ArduinoJson.h>
#include "Pins.h"
#include "sensors/units/UnitModel.h"
#include "sensors/ports/PortType.h"

#define DEFAULT_ROUTER_SSID "MisBKit00"
#define DEFAULT_ROUTER_PSWD "ensadmbk00"

struct Iconfiguration : public Printable{
  bool updated = false;
  virtual void add_to_json(JsonDocument& doc) = 0;
  virtual void set_from_json(const JsonDocument& doc) = 0;
  virtual size_t printTo(Print& p) const = 0;
};

struct kitConfiguration : Iconfiguration{
    const uint8_t v_maj = V_MAJ;
    const uint8_t v_min = V_MIN;
    const uint8_t v_patch = V_PATCH;
    
    void add_to_json(JsonDocument& doc) override {

    }

    String v_as_str() const {
      char buf[16];
      snprintf(buf, sizeof(buf), "%d.%d.%d", v_maj, v_min, v_patch);
      return String(buf);
    }


    void set_from_json(const JsonDocument& doc ) {

    }

    size_t printTo(Print& p) const {
        size_t n = 0;
      char buf[64];
      snprintf(buf, sizeof(buf), "Kit config:\n    Version: %s",
           v_as_str().c_str());
      n += p.print(buf);
        return n;
    }
};

struct networkConfiguration : Iconfiguration{
    char ssid[32]{DEFAULT_ROUTER_SSID};
    char pswd[64]{DEFAULT_ROUTER_PSWD};
    bool dhcp{false};
    IPAddress ip{192,168,0,125};

    void add_to_json(JsonDocument& doc) override {
      doc["ssid"] = ssid;
      doc["pswd"] = pswd;
      doc["dhcp"] = dhcp;
      doc["ip"][0] = ip[0];
      doc["ip"][1] = ip[1];
      doc["ip"][2] = ip[2];
      doc["ip"][3] = ip[3];
    }

    void set_from_json(const JsonDocument& doc ){
      const char* in_ssid = doc["ssid"] | ssid;
      const char* in_pswd = doc["pswd"] | pswd;
      snprintf(ssid, sizeof(ssid), "%s", in_ssid);
      snprintf(pswd, sizeof(pswd), "%s", in_pswd);

      dhcp = doc["dhcp"] | dhcp;

      JsonArrayConst ip_arr = doc["ip"].as<JsonArrayConst>();
      if (ip_arr.size() == 4) {
        ip[0] = ip_arr[0].as<uint8_t>();
        ip[1] = ip_arr[1].as<uint8_t>();
        ip[2] = ip_arr[2].as<uint8_t>();
        ip[3] = ip_arr[3].as<uint8_t>();
      }
    }

    size_t printTo(Print& p) const {
        size_t n = 0;
      char buf[160];
      snprintf(buf, sizeof(buf),
           "Net config:\n    ssid: %s pswd: %s\n    ip: %d.%d.%d.%d\n    dhcp : %d",
           ssid, pswd, ip[0], ip[1], ip[2], ip[3], dhcp);
      n += p.print(buf);
        return n;
    }
};

struct rcConfiguration : Iconfiguration {
  static constexpr uint8_t k_max_channels = 4U;
  static constexpr uint16_t k_default_min_angle = 0U;
  static constexpr uint16_t k_default_max_angle = 180U;
  static constexpr uint16_t k_default_speed_dps = 90U;
  static constexpr uint8_t k_default_easing_type = 0U;

  bool channel_enabled[k_max_channels]{false};
  uint16_t channel_min_angle[k_max_channels]{k_default_min_angle};
  uint16_t channel_max_angle[k_max_channels]{k_default_max_angle};
  uint16_t channel_default_speed_dps[k_max_channels]{k_default_speed_dps};
  uint8_t channel_easing_type[k_max_channels]{k_default_easing_type};

  rcConfiguration() {
    reset_defaults();
  }

  void reset_defaults() {
    for (uint8_t i = 0; i < k_max_channels; ++i) {
      channel_enabled[i] = false;
      channel_min_angle[i] = k_default_min_angle;
      channel_max_angle[i] = k_default_max_angle;
      channel_default_speed_dps[i] = k_default_speed_dps;
      channel_easing_type[i] = k_default_easing_type;
    }
  }

  void apply_channel_json(uint8_t index, JsonObjectConst channel) {
    channel_enabled[index] = channel["enabled"] | channel_enabled[index];
    channel_default_speed_dps[index] = channel["default_speed_dps"] | channel_default_speed_dps[index];

    const uint16_t min_angle = channel["min_angle"] | channel_min_angle[index];
    const uint16_t max_angle = channel["max_angle"] | channel_max_angle[index];
    if (min_angle > max_angle) {
      channel_min_angle[index] = max_angle;
      channel_max_angle[index] = min_angle;
    } else {
      channel_min_angle[index] = min_angle;
      channel_max_angle[index] = max_angle;
    }

    channel_easing_type[index] = channel["easing_type"] | channel_easing_type[index];
  }

  void add_to_json(JsonDocument& doc) override {
    JsonObject rc = doc["rc"].to<JsonObject>();
    JsonArray channels_json = rc["channels"].to<JsonArray>();
    for (uint8_t i = 0; i < k_max_channels; ++i) {
      JsonObject channel = channels_json.add<JsonObject>();
      channel["enabled"] = channel_enabled[i];
      channel["min_angle"] = channel_min_angle[i];
      channel["max_angle"] = channel_max_angle[i];
      channel["default_speed_dps"] = channel_default_speed_dps[i];
      channel["easing_type"] = channel_easing_type[i];
    }
  }

  void set_from_json(const JsonDocument& doc) override {
    reset_defaults();

    JsonObjectConst rc = doc["rc"].as<JsonObjectConst>();
    if (rc.isNull()) {
      updated = true;
      Serial.println("RC is null");
      return;
    }


    JsonArrayConst channels_json = rc["channels"].as<JsonArrayConst>();
    if (!channels_json.isNull()) {
      uint8_t fallback_index = 0U;
      for (JsonVariantConst channel_variant : channels_json) {
        if (fallback_index >= k_max_channels) {
          break;
        }

        JsonObjectConst channel = channel_variant.as<JsonObjectConst>();
        if (channel.isNull()) {
          ++fallback_index;
          continue;
        }

        uint8_t target_index = channel["channel"] | fallback_index;
        if (target_index >= k_max_channels) {
          target_index = fallback_index;
        }

        apply_channel_json(target_index, channel);
        ++fallback_index;
      }
    }

    updated = true;
  }

  size_t printTo(Print& p) const override {
    size_t n = 0;
    char buf[256];

    n += p.print("RC config:\n");
    for (uint8_t i = 0; i < k_max_channels; ++i) {
      snprintf(buf,
               sizeof(buf),
               "    channel %u: enabled=%d min=%u max=%u speed=%u easing=%u\n",
               static_cast<unsigned>(i),
               channel_enabled[i],
               static_cast<unsigned>(channel_min_angle[i]),
               static_cast<unsigned>(channel_max_angle[i]),
               static_cast<unsigned>(channel_default_speed_dps[i]),
               static_cast<unsigned>(channel_easing_type[i]));
      n += p.print(buf);
    }
    return n;
  }
};

struct sensorConfiguration : Iconfiguration {

  // Allow multiple units per physical connector and explicit port binding.
  static constexpr uint8_t k_max_units_per_port = 8;

  // struct UnitSpec {
  //   UnitModel model{UnitModel::none};
  //   bool enabled{false};
  //   float alpha{0.0f};
  //   int addr{-1};
  // };

  // Explicit port type enum (none/portA/portB/portC).
  PortType port_binding[MBK_PORT_COUNT]{};

  // Multiple units storage per port.
  UnitSpec port_units[MBK_PORT_COUNT][k_max_units_per_port]{};
  uint8_t port_unit_count[MBK_PORT_COUNT]{};

  sensorConfiguration() {
    for (size_t i = 0; i < MBK_PORT_COUNT; i++) {
      port_binding[i] = PortType::none;
      port_unit_count[i] = 0;
      for (size_t j = 0; j < k_max_units_per_port; j++) {
        port_units[i][j] = UnitSpec();
      }
    }
  }

  void clear_units(int port) {
      port_unit_count[port] = 0;
      for (size_t j = 0; j < k_max_units_per_port; ++j) {
        port_units[port][j] = UnitSpec();
      }
  }

  void add_unit_to_port(uint8_t port_idx, UnitSpec specs){
    if(port_unit_count[port_idx] < k_max_units_per_port){
      uint8_t count = port_unit_count[port_idx];
      port_units[port_idx][count].model = specs.model;
      port_units[port_idx][count].enabled = specs.enabled;    // default: disabled
      port_units[port_idx][count].alpha = specs.alpha;    // default: 0.0
      port_units[port_idx][count].addr = -1;        // default: -1 meaning "not set"
      port_unit_count[port_idx]++;

    }
  }

  void add_to_json(JsonDocument& doc) override {
    JsonArray ports = doc["ports"].to<JsonArray>();
    for (size_t i = 0; i < MBK_PORT_COUNT; i++) {
      JsonObject portObj = ports.add<JsonObject>();
      portObj["type"] = port_type_to_string(port_binding[i]);
      JsonArray unitsArr = portObj["units"].to<JsonArray>(); 
      for (uint8_t u = 0; u < port_unit_count[i]; u++) {
        JsonObject unitObj = unitsArr.add<JsonObject>();
        unitObj["model"] = model_to_str(port_units[i][u].model);
        unitObj["io"] = static_cast<int>(port_units[i][u].enabled);
        unitObj["alpha"] = port_units[i][u].alpha;
        if (port_units[i][u].addr >= 0) unitObj["addr"] = port_units[i][u].addr;
      }
    }
  }

  void set_from_json(const JsonDocument& doc) override {
    // Top-level guard: expect a JSON array at `ports`.
    // .is<JsonArray>() always return false so we have to check for a JsonArrayConst
    if (!doc["ports"].is<JsonArrayConst>()) {
      Log.warningln("sensorConfiguration.set_from_json: missing or invalid ports array");
      for (size_t i = 0; i < MBK_PORT_COUNT; i++) {
        port_unit_count[i] = 0;                 // no units on this port
        port_binding[i] = PortType::none; // no binding/resource required
      }
      //Log.infoln("sensorConfiguration.set_from_json: exit invalid payload");
      return;
    }

    // Read the ports array and remember how many entries were provided.
    // The device has a fixed number of physical connectors (MBK_PORT_COUNT);
    // the JSON array may be shorter (missing ports) — handle that gracefully.
    JsonArrayConst ports = doc["ports"].as<JsonArrayConst>();
    size_t n = ports.size();

    for (size_t i = 0; i < MBK_PORT_COUNT; i++) {
      // If the JSON provides fewer port entries than the hardware supports,
      // treat the remaining ports as empty/disabled.
      if (i >= n) {
        port_unit_count[i] = 0;
        port_binding[i] = PortType::none;
        continue;
      }

      // .is<JsonObject>() always return false so we have to check for a JsonObjectConst
      JsonObjectConst portObj = ports[i].as<JsonObjectConst>();

      
      
      // Require an exact textual match for the binding. Accepted values are
      // exactly "portA", "portB", or "portC". Anything else leaves the binding as `none` 
      const char* type_str = portObj["type"] | "";
      port_binding[i] = PortType::none;
      if (type_str && type_str[0] != '\0') {
        port_binding[i] = port_type_from_string(type_str);
        if (port_binding[i] == PortType::none) {
          Log.warningln("sensorConfiguration.set_from_json: invalid port type '%s' for port %u", type_str, (unsigned)i);
        }
      }

      // Each entry in `units` is an object describing a single logical device attached to that physical connector. 
      if (portObj["units"].is<JsonArrayConst>()) {
        JsonArrayConst units = portObj["units"].as<JsonArrayConst>();
        size_t units_n = units.size();

        uint8_t count = 0;
        for (JsonVariantConst u : units) {
          // Enforce a small, fixed upper bound to avoid unbounded memory usage.
          if (count >= k_max_units_per_port) {
            Log.warningln("port[%u] - reached k_max_units_per_port (%u), skipping remaining units", (unsigned)i, (unsigned)k_max_units_per_port);
            break;
          }

          const char* model_str = u["model"] | "";
          port_units[i][count].model = model_str_to_enum(model_str);
          port_units[i][count].enabled = u["io"].as<bool>() | false;    // default: disabled
          port_units[i][count].alpha = u["alpha"] | 0.0f;    // default: 0.0
          port_units[i][count].addr = u["addr"] | -1;        // default: -1 meaning "not set"
          // Log.infoln("sensorConfiguration.set_from_json: port %u unit %u model=%s enabled=%u alpha=%.3f addr=%d", (unsigned)i, (unsigned)count, model_str, port_units[i][count].enabled ? 1 : 0, port_units[i][count].alpha, port_units[i][count].addr);
          count++;
        }
        port_unit_count[i] = count; // number of units configured on this port
      } else {
        // No units array => explicitly empty port configuration
        port_unit_count[i] = 0;
        Log.infoln("sensorConfiguration.set_from_json: port %u has no units array", (unsigned)i);
      }

    }
    updated = true;
    }

  size_t printTo(Print& p) const {
    size_t n = 0;
    char buf[256];

    n += p.print("Sensor configuration:\n");
    for (size_t i = 0; i < MBK_PORT_COUNT; i++) {
      const char* type_str = port_type_to_string(port_binding[i]);
      snprintf(buf, sizeof(buf), "  Port %u: type=%s units=%u\n", (unsigned)i, type_str, (unsigned)port_unit_count[i]);
      n += p.print(buf);

      for (uint8_t u = 0; u < port_unit_count[i]; u++) {
        const UnitSpec& us = port_units[i][u];
        const char* model = model_to_str(us.model);
        const char* addr_str = (us.addr >= 0) ? "set" : "none";
        // Print per-unit line with model, enabled, alpha and address presence
        snprintf(buf, sizeof(buf), "    [%u] model=%s enabled=%d alpha=%.3f addr=%s\n", (unsigned)u, model, us.enabled ? 1 : 0, us.alpha, addr_str);
        n += p.print(buf);
      }
      if (port_unit_count[i] == 0) {
        n += p.print("    (no units)\n");
      }
    }

    return n;
  }
};
#endif // MBK_CONFIGURATION_H