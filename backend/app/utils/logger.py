"""
Logging Configuration
Sets up logging for the application.
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.app.config import config


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up application logging.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        use_colors: Use colored output for console
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("agentic_rag")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if use_colors:
        console_format = ColorFormatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        
        file_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (uses module name if None)
    
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"agentic_rag.{name}")
    return logging.getLogger("agentic_rag")


# Convenience functions
def log_info(message: str, **kwargs):
    """Log an info message."""
    logger = get_logger()
    logger.info(message, **kwargs)


def log_error(message: str, exc_info: bool = False, **kwargs):
    """Log an error message."""
    logger = get_logger()
    logger.error(message, exc_info=exc_info, **kwargs)


def log_warning(message: str, **kwargs):
    """Log a warning message."""
    logger = get_logger()
    logger.warning(message, **kwargs)


def log_debug(message: str, **kwargs):
    """Log a debug message."""
    logger = get_logger()
    logger.debug(message, **kwargs)


class MetricLogger:
    """Logger specifically for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger("metrics")
        self.metrics: list = []
    
    def log_latency(self, operation: str, latency: float, **metadata):
        """Log operation latency."""
        self.logger.info(f"LATENCY | {operation} | {latency:.3f}s | {metadata}")
        self.metrics.append({
            "type": "latency",
            "operation": operation,
            "value": latency,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        })
    
    def log_throughput(self, operation: str, count: int, duration: float, **metadata):
        """Log throughput metric."""
        rate = count / duration if duration > 0 else 0
        self.logger.info(f"THROUGHPUT | {operation} | {rate:.2f}/s | {metadata}")
        self.metrics.append({
            "type": "throughput",
            "operation": operation,
            "count": count,
            "duration": duration,
            "rate": rate,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        })
    
    def log_error_rate(self, operation: str, errors: int, total: int, **metadata):
        """Log error rate metric."""
        rate = errors / total if total > 0 else 0
        self.logger.info(f"ERROR_RATE | {operation} | {rate:.2%} | {metadata}")
        self.metrics.append({
            "type": "error_rate",
            "operation": operation,
            "errors": errors,
            "total": total,
            "rate": rate,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        })
    
    def get_recent_metrics(self, n: int = 100) -> list:
        """Get recent metrics."""
        return self.metrics[-n:]


# Global metric logger
_metric_logger: Optional[MetricLogger] = None


def get_metric_logger() -> MetricLogger:
    """Get the global metric logger instance."""
    global _metric_logger
    if _metric_logger is None:
        _metric_logger = MetricLogger()
    return _metric_logger


# Initialize logging on import
_logger = setup_logging(level="INFO")
