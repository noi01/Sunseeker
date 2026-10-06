import threading
from typing import Callable, Optional

import websocket


class RLWebSocketClient:
    def __init__(
        self,
        url: str,
        on_message: Optional[Callable[[str], None]] = None,
        on_open: Optional[Callable[[], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_close: Optional[Callable[[int, str], None]] = None,
    ):
        self.url = url
        self.on_message_callback = on_message
        self.on_open_callback = on_open
        self.on_error_callback = on_error
        self.on_close_callback = on_close
        self._socket_app = None
        self._thread = None
        self._connected_event = threading.Event()

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()
    
    def log(self, mess):
        print("[WS] {}".format(mess))

    def connect(self, timeout: float = 5.0) -> bool:
        if self._thread and self._thread.is_alive():
            return self.is_connected

        self._connected_event.clear()

        self._socket_app = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )

        self._thread = threading.Thread(
            target=self._socket_app.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True,
        )
        self._thread.start()
        success = False
        if not self._connected_event.wait(timeout=timeout):
             self.log("Connection timeout for {}".format(self.url))
        else:
            success = True
        return success

    def close(self, timeout: float = 2.0):
        if self._socket_app is not None:
            self._socket_app.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def send(self, message: str):
        if self._socket_app is None or self._socket_app.sock is None:
            raise RuntimeError("WebSocket is not connected")
        print(message)
        self._socket_app.send(message)

    def _on_open(self, ws):
        self._connected_event.set()
        self.log("Connected to {}".format(self.url))
        if self.on_open_callback is not None:
            self.on_open_callback()

    def _on_message(self, ws, message):
        self.log("Received : {}".format(message))
        if self.on_message_callback is not None:
            self.on_message_callback(ws, message)

    def _on_error(self, ws, error):
        self.log("Error {}".format(error))
        if self.on_error_callback is not None:
            self.on_error_callback(error)

    def _on_close(self, ws, close_status_code, close_msg):
        self._connected_event.clear()
        self.log("Closed (code={}, msg={})".format(close_status_code, close_status_code))
        if self.on_close_callback is not None:
            self.on_close_callback(close_status_code, close_msg)
