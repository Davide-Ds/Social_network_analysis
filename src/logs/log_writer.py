# logutils.py
import sys
import os
from datetime import datetime

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def setup_logging(log_dir="src/logs"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"[LOG] Logging started: {log_path}")
