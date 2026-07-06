/**
 * @file MisBKit.cpp
 * @author Etienne Montenegro
 * @brief Implementation file for MisBKit.h
 */

#include "MisBKit.h"

namespace {
  static uint32_t update_fps_last_report_ms = 0;
  static uint32_t update_fps_frame_count = 0;
  static constexpr uint32_t UPDATE_FPS_REPORT_INTERVAL_MS = 1000;

  static void report_update_fps() {
    update_fps_frame_count++;
    uint32_t now = millis();
    if (update_fps_last_report_ms == 0) {
      update_fps_last_report_ms = now;
      return;
    }

    uint32_t elapsed = now - update_fps_last_report_ms;
    if (elapsed >= UPDATE_FPS_REPORT_INTERVAL_MS) {
      static u8_t count = 0;
      
      float fps = update_fps_frame_count * 1000.0f / elapsed;
      if(count >= 10){
        Log.infoln("update() fps: %F", fps);
        count = 0;
      }
      
      update_fps_frame_count = 0;
      update_fps_last_report_ms = now;
      count++;
    }
  }
}


void MisBKit::initialize(bool _clear) {
  Log.infoln("Starting MisBKit initialization");
  // if(_clear) config_manager.clear();
 
  if(config_manager.initialize()){
    load_configurations();
    config_manager.close();
  }
  
  uint8_t id = net_config.ip[3];
  network_manager.update_ap_credentials(id);
  network_manager.initialize();

  Log.warningln("network initialized");

  
  ws.setConnectionHandler([this](IPAddress ip) { this->pair(ip); });
  ws.setMessageHandler([this](const Command& cmd){this->process_command(cmd);});
  ws.setDisconnectionHandler([this](){this->disconnect();});
  ws.initialize();

  web_interface.initialize();
  server.begin();
  
  sensor_manager.initialize();

  motor_manager.initialize();
  // // led::init();
  // // // battery pin configuration
  // // pinMode(pins::battery, INPUT);

  ws.setMotorManager(&motor_manager);



  Log.infoln("MisBKit initialization done.");
}

void MisBKit::load_configurations(){
  if(!config_manager.load()) return;

  JsonDocument& doc = config_manager.get_doc();
  kit_config.set_from_json(doc);
  Log.infoln("%p", kit_config);
  net_config.set_from_json(doc);
  Log.infoln("%p", net_config);
  sensor_config.set_from_json(doc);
  Log.infoln("%p", sensor_config);
  rc_config.set_from_json(doc);
  Log.infoln("%p", rc_config);
  sensor_config.updated = false;
}

void MisBKit::save_configuration(){
  kit_config.add_to_json(config_manager.get_doc());
  net_config.add_to_json(config_manager.get_doc());
  sensor_config.add_to_json(config_manager.get_doc());
  rc_config.add_to_json(config_manager.get_doc());
  if(config_manager.save())
    Log.infoln("Successfully savec configuration");
}


void MisBKit::execute_action(JsonObject action){

    switch (Command::to_enum(action)){
      case CMD_REBOOT:
        reboot_flag = true;
        break;
  
      case CMD_SCAN:
        scan_flag = true;
        break;
  
      case CMD_STOP: {
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr) m->stop();
        break;
      }
  
      case CMD_STOP_ALL: {
        motor_manager.stopAll();
        break;
      }



      case CMD_MOTOR_ID:{
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr){
            m->setID(action["val"].as<int>()); 
        }
        break;
      }

      case CMD_MOTOR_FACTORY_RESET:{
        motor_manager.factory_reset();
        break;
      }

      case CMD_DEEP_SCAN:{
        motor_manager.reconfigureBadBaud();
        scan_flag = true;

        break;
      }

      case CMD_CHANGE_MOTOR_BAUD:{
        motor_manager.setControlBaud( action["val"].as<int>() );
        motor_manager.reconfigureBadBaud();
        scan_flag = true;

        break;
      }

 

      case CMD_SET_MODE:{
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr){
           if(action["val"].as<int>()){
            m->setMode(WHEEL_MODE);
           }else{
            m->setMode(JOINT_MODE);
           }
        }
        break;
      }

      case CMD_SET_STIFF:{
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr){
           if(action["val"].as<int>()){
            m->setTorqueOn();
           }else{
            m->setTorqueOff();
           }
        }
        break;
      }
      
      
      

      case CMD_WHEEL: {
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr) m->setWheelSpeed(action["val"].as<uint16_t>());
        break;
      }
  
      case CMD_JOINT: {
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr) m->setGoalPosition(action["val"].as<uint16_t>());
        break;
      }
  
      

      case CMD_JOINT_VELOCITY: {
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr) m->setJointSpeed(action["val"].as<uint16_t>());
        break;
      }
  
      case CMD_STATUS_DUMP: {
        Motor *m = motor_manager.get(action["id"].as<uint8_t>());
        if (m != nullptr) m->updateInfos();
        break;
      }
  
      case CMD_SENSOR_CONFIG: {
        JsonDocument doc = action["val"];
        sensor_config.set_from_json(doc);
        break;
      }
      
      case CMD_SENSOR_DATA:{
      if(!sensor_manager.no_sensor_enabled()){
          JsonDocument sensor_data;
          sensor_manager.add_data_to_json(sensor_data);
          ws.reply("sensordata",sensor_data);
        } 
      }
        break;

      case CMD_SENSOR_SCAN: {
        uint8_t port_id = action["id"];
        sensor_manager.scan(port_id);
        JsonDocument reply;
        sensor_config.add_to_json(reply);
        rc_config.add_to_json(reply);
        ws.reply("sensorconfig", reply);
        break;
      }
  
      case CMD_SAVE_SENSOR_CONFIG:
        Log.traceln("Saving sensor config");
        save_configuration();
        network_manager.remove_wifi_events();  // prevent going in the reconnecting loop
        reboot_flag = true;
        break;

      case CMD_SYNC_WHEEL:{
        const uint8_t len = action["id"].size();
        uint8_t id_list[len];
        copyArray(action["id"],id_list,len);
        motor_manager.set_wheel_speed_sync(id_list,len,action["val"].as<uint16_t>());
        break;
      }

      case CMD_SYNC_JOINT:{
        const uint8_t len = action["id"].size();
        uint8_t id_list[len];
        copyArray(action["id"],id_list,len);
        motor_manager.set_goal_position_sync(id_list,len,action["val"].as<uint16_t>());
        break;
      }

      case CMD_SYNC_JOINT_VELOCITY:{
        const uint8_t len = action["id"].size();
        uint8_t id_list[len];
        copyArray(action["id"],id_list,len);
        motor_manager.set_joint_speed_sync(id_list,len,action["val"].as<uint16_t>());
        break;
      }
      
      case CMD_SYNC_STOP:{
        const uint8_t len = action["id"].size();
        uint8_t id_list[len];
        copyArray(action["id"],id_list,len);
        motor_manager.stop_sync(id_list,len);
        break;
      }


      
      default:
        Log.errorln("Command does not exist");
        break;
    }
}


void MisBKit::process_command(const Command& c) {
  // Log.traceln("Processing command:");
  JsonDocument doc;
  if (c.is_valid(doc)) {
    // Log.traceln("Command is valid");
    if (c.is_single_command(doc)) {
      // Log.traceln("Executing single command");
      execute_action(doc.as<JsonObject>());
    } else {
      // Log.traceln("Executing multiple commands");
      if (doc["cmds"].is<JsonArray>()) {
        for (int i = 0; i < doc["cmds"].size(); i++) {
          // Log.traceln("Executing command %d/%d", i+1, doc["cmds"].size());
          execute_action(doc["cmds"][i]);
        }
      }
    }
  } else {
    Log.errorln("Command is invalid");
  }
}


void MisBKit::update() {
  ws.update();
  sensor_manager.update();

  if (scan_flag) {
    uint8_t found = motor_manager.scanMotors();
    Log.info("Found %d motor during scan", found);
    JsonDocument doc;
    doc["ids"].to<JsonArray>();
    if (found > 0) {

      for (Motor *m : motor_manager.motors()) {
        if (m) doc["ids"].add<int>(m->getID());
      }
    }

    ws.reply("scan",doc);

    scan_flag = false;
  }

  if (reboot_flag) {
    reboot();
  }

}


void MisBKit::pair(IPAddress ip) {
  Log.traceln("Trying to pair mcu");

  MisBKit::connected = true;
  
  JsonDocument reply;
  sensor_config.add_to_json(reply);
  ws.reply("sensorconfig", reply);

  
  
  JsonDocument mess;
  mess["ip"] = ip.toString().c_str();
  mess["version"] = kit_config.v_as_str();
  ws.reply("pair", mess);

  Log.traceln("End of pairing");
}

void MisBKit::reboot() { 
  Log.warningln("Rebooting");
  delay(1000);
  ESP.restart(); 
}
