/**
 * @file MisBKit.h
 * @author Etienne Montenegro
 * @brief This file contains all the methods and variables related to a MisBKit
 * configuration and status
 */
#ifndef MISBKIT_H
#define MISBKIT_H

#include <Arduino.h>
#include <ArduinoJson.h>
#include <ArduinoLog.h>



#include "Configuration.h"
#include "PersistantConfiguration.h"
#include "NetworkManager.h"
#include "motors/MotorManager.h"
#include "sensors/sensorManager.h"
#include "WebInterfaceClass.h"
#include "Websocket.h"


#define WS_NBR_MAX_MOTORS 6
#define WS_NBR_MAX_CURVE_POINT 500

#define WS_NBR_MAX_SENSORS 10



class MisBKit{
  kitConfiguration kit_config;
  networkConfiguration net_config;
  sensorConfiguration sensor_config;
  rcConfiguration rc_config;
  PersistantConfiguration config_manager;

  NetworkHandler network_manager{&net_config};
  AsyncWebServer server{80};
  WebInterface web_interface{&server};
  WebSocketProtocol ws{&server};

  MotorManager motor_manager;
  SensorManager sensor_manager{&sensor_config};



  

  bool connected{false};
  bool scan_flag = false;
  bool reboot_flag = false;

  void load_configurations();
  void save_configuration();
  void reboot();

  void pair(IPAddress ip);
  inline void disconnect(){connected = false;}

  //MisBKit::Commands cmd_str_to_enum(const char *cmd_str); // move this to the Command struct so a command

  //maybe have an abstract interface to buffer handle and parse so that ws can be replaced easily.
  // the Misbkit should have a function that receives a MisBKit::Command_t as an argument to execute an action
  void process_command(const Command& c);
  void reply_all(const char* cmd, JsonDocument& val);
  // void parseCommand(const JsonObject cmd);
  void execute_action(JsonObject action);

  
  public:
  MisBKit(){

  }
  void initialize(bool clear);
  void update();





};

#endif