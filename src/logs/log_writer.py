# logutils.py
import sys
import os
from datetime import datetime

class DualLogger:
    """
    Logger that writes to both terminal and a log file.
    """
    def __init__(self, filename):
        """
        Initialize the dual logger.
        Args:
            filename (str): Path to the log file.
        """
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        """ Write message to both terminal and log file.
        Args:
            message (str): The message to log.
        """
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        """ Flush both terminal and log file. """
        self.terminal.flush()
        self.log.flush()

def setup_logging(log_dir="src/logs"):
    """
    Set up logging to both terminal and a log file.
    Args:
        log_dir (str): Directory to store log files.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(log_dir, f"log_{timestamp}.txt")
    logger = DualLogger(log_path)
    sys.stdout = logger
    sys.stderr = logger
    print(f"[LOG] Logging started: {log_path}")
