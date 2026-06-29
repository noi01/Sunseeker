import json
import threading
from typing import Optional

import websocket


class MisBKit:
    def __init__(self):
        self._paired = False
        self._version = "0.0.0"
        self._ws_url = "ws://192.168.0.125/ws"
        self._socket_app = None
        self._ws_thread = None
        self._connected_event = threading.Event()
        self.timeout = 5
        self._handle_sensor_data = None
    @property
    def paired(self):
        """Get the paired state."""
        return self._paired

    @paired.setter
    def paired(self, value):
        """Set the paired state."""
        if not isinstance(value, bool):
            raise ValueError("paired must be a boolean value")
        self._paired = value

    @property
    def version(self):
        """Get the version of the kit."""
        return self._version

    @version.setter
    def version(self, value):
        if not isinstance(value, str):
            raise ValueError("Version number must be string")
        self._version = value

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    def _ws_log(self, message):
        print("[WS] {}".format(message))

    def connect(self) -> bool:
        if self._ws_thread and self._ws_thread.is_alive():
            print("here")
            return self.is_connected

        self._connected_event.clear()

        self._socket_app = websocket.WebSocketApp(
            self._ws_url,
            on_open=self._handle_ws_open,
            on_message=self._handle_ws_message,
            on_error=self._handle_ws_error,
            on_close=self._handle_ws_close,
        )

        self._ws_thread = threading.Thread(
            target=self._socket_app.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        self._ws_thread.start()
        if not self._connected_event.wait(self.timeout):
            self._ws_log("Connection timeout for {}".format(self._ws_url))
            return False
        return True
    

    def close(self, timeout: float = 2.0):
        if self._socket_app is not None:
            self._socket_app.close()
        if self._ws_thread is not None and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=timeout)

    def send(self, message: str):
        if self._socket_app is None or self._socket_app.sock is None:
            raise RuntimeError("WebSocket is not connected")
        print(message)
        self._socket_app.send(message)

    def request_sensor_data(self):
        if self.paired and self.is_connected:
            self.send(self._make_command("sensordata"))
        
    def _handle_ws_open(self, ws):
        _ = ws
        self._connected_event.set()
        self._ws_log("Connected to {}".format(self._ws_url))
        self._on_open_callback()

    def _handle_ws_message(self, ws, message):
        self._ws_log("Received : {}".format(message))
        msg_json = json.loads(message)
        reply_cmd = msg_json["reply"]
        print("command:", reply_cmd, type(reply_cmd))
        match reply_cmd:
            case "pair":
                print(
                    "Paired to kit {} version {}".format(
                        msg_json["val"]["ip"], msg_json["val"]["version"]
                    )
                )
                self.paired = True
                self.version = msg_json["val"]["version"]
                return
            case "sensorconfig":
                print("Received sensor configuration {}".format(msg_json["val"]))
                return
            case "sensordata":
                if self._handle_sensor_data == None:
                    print(msg_json["val"])
                else:
                    self._handle_sensor_data(msg_json["val"])
                return
        print("leaving callback")

    def _handle_ws_error(self, ws, error):
        _ = ws
        self._ws_log("Error {}".format(error))
        self._on_error_callback(error)

    def _handle_ws_close(self, ws, close_status_code, close_msg):
        _ = ws
        self._connected_event.clear()
        self._ws_log("Closed (code={}, msg={})".format(close_status_code, close_status_code))
        self._on_close_callback(close_status_code, close_msg)

   

    def _on_open_callback(self):
        print("Socket Open")

    def _on_error_callback(self, error):
        _ = error

    def _on_close_callback(self, close_status_code, close_msg):
        _ = close_status_code
        _ = close_msg

    

    def close_websocket_client(self):
        self.close()

    def _make_command(self, name, id = None, val = None):
        mess ={}
        if id == None and val == None:
                mess = {
                    "cmd": name
                }
        elif val == None:
            mess = {
                "cmd": name,
                "id": id
            } 
        elif id == None:
            mess = {
                "cmd": name,
                "val": val
            } 
        else:
            mess = {
                "cmd": name,
                "id": id,
                "val": val
            }

        return json.dumps(mess)