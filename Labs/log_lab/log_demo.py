import logging
import json
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Custom JSON formatter
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line_number": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


# Configure logging with both console and file handlers
def setup_logging():
    # Create logger
    logger = logging.getLogger("MLOps_Logger")
    logger.setLevel(logging.DEBUG)
    
    # Console Handler - Human readable format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File Handler - JSON format
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


# Demo function to show different log levels
def demo_logging():
    logger = setup_logging()
    
    logger.info("Application started - Priyanka Custom Logging Demo")
    logger.debug("This is a debug message with detailed information")
    logger.warning("This is a warning - something might need attention")
    logger.error("This is an error message")
    
    # Simulate processing some data
    try:
        data = [1, 2, 3, 4, 5]
        result = sum(data) / len(data)
        logger.info(f"Data processing completed. Average: {result}")
    except Exception as e:
        logger.exception("An error occurred during data processing")
    
    # Simulate an error
    try:
        result = 10 / 0
    except ZeroDivisionError:
        logger.exception("Division by zero error caught and logged")
    
    logger.info("Application finished successfully")


if __name__ == "__main__":
    demo_logging()