import sys
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.logger import get_logger

WATCH_DIR = "/app/data/new"
logger = get_logger(__name__)


class NewDataHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Handle new data files by triggering the training pipeline."""
        logger.info(f"[Watcher] Detected new file: {event.src_path}")
        if event.src_path.endswith(".csv"):
            logger.info(f"[Watcher] New data detected: {event.src_path}")
            subprocess.Popen([
                "python", "src/run_training_pipeline.py", event.src_path
            ])


if __name__ == "__main__":
    os.makedirs(WATCH_DIR, exist_ok=True)
    event_handler = NewDataHandler()
    observer = PollingObserver()
    observer.schedule(event_handler, path=WATCH_DIR, recursive=False)
    observer.start()
    logger.info(f"[Watcher] Watching absolute path: {os.path.abspath(WATCH_DIR)}")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
