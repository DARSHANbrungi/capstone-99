import logging
import os
from datetime import datetime

# Define the name for the log file using the current timestamp
LOG_FILE_NAME = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"

# Define the directory to store the log files
LOGS_DIRECTORY = os.path.join(os.getcwd(), "logs")

# Create the logs directory if it doesn't already exist
os.makedirs(LOGS_DIRECTORY, exist_ok=True)

# Define the full path for the log file
LOG_FILE_PATH = os.path.join(LOGS_DIRECTORY, LOG_FILE_NAME)

# Configure the root logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    # Define the format for the log messages
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Example of how to use the logger:
# if __name__ == "__main__":
#     logging.info("Logging has started.")

