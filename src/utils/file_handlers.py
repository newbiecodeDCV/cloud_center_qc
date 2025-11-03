import os
import time
import threading
import hashlib
import logging
from watchdog.events import FileSystemEventHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 3.0


class CSVWatcher(FileSystemEventHandler):
    def __init__(self,
                 csv_path: str,
                 db_path: str,
                 reload_callback,
                 debounce_seconds=DEBOUNCE_SECONDS):
        self.csv_path = csv_path
        self.db_path = db_path
        self.reload_callback = reload_callback
        self.debounce_seconds = debounce_seconds
        self._last_trigger_time = 0
        self._lock = threading.Lock()

    def on_modified(self, event):
        if not event.src_path.endswith(self.csv_path):
            return

        now = time.time()
        with self._lock:
            if now - self._last_trigger_time < self.debounce_seconds:
                return
            self._last_trigger_time = now

        logging.info("Detected change in CSV file. Reloading database...")
        threading.Thread(target=self._safe_reload,
                         args=(self.csv_path, self.db_path),
                         daemon=True).start()

    def _safe_reload(self,
                     csv_path: str,
                     db_path: str):
        try:
            time.sleep(self.debounce_seconds)  # wait until changes settle
            self.reload_callback(csv_path, db_path)
        except Exception as e:
            logging.error(f"Error reloading DB: {e}")
