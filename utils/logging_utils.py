import logging
import pycolmap
import sys
from pathlib import Path
from datetime import datetime

def configure_logging(output_dir):
    """Configure pycolmap logging system and file logging"""
    
    # Create the logs directory
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"colmap_log_{timestamp}.txt"
    
    # Configure pycolmap logging
    pycolmap.logging.logtostderr = True          # Output logs to console
    pycolmap.logging.verbose_level = 4           # Set maximum verbosity level
    pycolmap.logging.alsologtostderr = True
    pycolmap.logging.minloglevel = pycolmap.logging.Level.INFO.value

    # Add a filter for Pillow logs
    class PillowFilter(logging.Filter):
        def filter(self, record):
            # Filter out debug-level logs from Pillow
            if record.name.startswith('PIL.'):
                return record.levelno >= logging.WARNING
            return True    
    
    # Create a file log handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Create a log formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(PillowFilter())
    
    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    
    # Add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PillowFilter())
    root_logger.addHandler(console_handler)

    # Also set Pillow logger level
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    logging.info(f"Log file created: {log_file}")
    return log_file
