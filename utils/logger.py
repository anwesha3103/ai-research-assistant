# ============================================================
# utils/logger.py — Logging Setup
# ============================================================
# Loguru instead of Python's built-in logging.
#
#  WHY LOGURU?
# ───────────────
# Built-in logging:
#   import logging
#   logging.basicConfig(level=logging.INFO)
#   logging.info("message")   → plain, hard to configure
#
# Loguru:
#   from loguru import logger
#   logger.info("message")    → colored, formatted, simple
#
# Loguru automatically shows:
#   • Timestamp
#   • Log level (INFO, WARNING, ERROR)
#   • File + line number where the log came from
#   • The message
#
# Example output:
#   2024-03-16 10:23:11 | INFO | loader.py:45 - Loaded 3 pages from doc.pdf

from loguru import logger
import sys
import os

# Remove default logger
logger.remove()

# ── Console Logger ───────────────────────────────────────────
# Colorized output in your VS Code terminal
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{file}:{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
)

# ── File Logger ──────────────────────────────────────────────
# Saves logs to a file — useful for debugging after the fact
os.makedirs("logs", exist_ok=True)

logger.add(
    "logs/app.log",
    rotation="10 MB",       # start a new file after 10MB
    retention="7 days",     # delete logs older than 7 days
    compression="zip",      # compress old log files
    level="DEBUG",          # log everything to file
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {file}:{line} - {message}",
)


def get_logger():
    """Return the configured logger. Import this in every module."""
    return logger
