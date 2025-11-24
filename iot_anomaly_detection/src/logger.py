import logging
import os
from datetime import datetime

# --- Log File Configuration ---
LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOGS_DIRECTORY = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't exist
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# Full path for the log file
LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

# --- Configure the Root Logger ---
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Optional: You can add a StreamHandler to also print logs to the console
# This can be helpful for real-time debugging.
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)
# console_handler.setFormatter(logging.Formatter("[ %(asctime)s ] %(levelname)s - %(message)s"))
# logging.getLogger().addHandler(console_handler)