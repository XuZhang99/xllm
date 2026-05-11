import logging
import os
from datetime import datetime

class GlogStyleFormatter(logging.Formatter):
    LEVEL_MAP = {
        logging.INFO: "I",
        logging.WARNING: "W",
        logging.ERROR: "E",
        logging.FATAL: "F",
        logging.DEBUG: "D",
    }

    def format(self, record):
        level = self.LEVEL_MAP.get(record.levelno, "I")

        now = datetime.fromtimestamp(record.created)
        timestamp = now.strftime("%Y%m%d %H:%M:%S")
        microsecond = f"{now.microsecond:06d}"

        pid = os.getpid()

        filename = record.filename
        lineno = record.lineno

        prefix = (
            f"{level}"
            f"{timestamp}.{microsecond} "
            f"{pid} "
            f"{filename}:{lineno}]"
        )

        message = record.getMessage()

        return f"{prefix} {message}"


logger = logging.getLogger("xllm")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(GlogStyleFormatter())

logger.addHandler(handler)
