import logging
import sys
import time
from pathlib import Path


def get_logger(
    log_level: int = 10,
    log_out: str = "console",
    name: str = "log",
    timedatelog: bool = False,
    log_path: Path | None = None,
    mode: str = "w",
) -> logging.Logger:
    """
    Generate logger object

    :param name: Name of logger
    :param log_level: Log level
    :param log_out: Whether to output log to 'file', 'console' or 'both'
    :param timedatelog: Whether to append datetime to log filename
    :param log_path: File to log to if log_out is 'file' or 'both'. Ignored otherwise.
                     Pass either this or log_dir
    :param mode: Mode to open log file as.
    :return: logging.Logger object
    """
    if log_out not in ("file", "both", "console", None):
        raise ValueError(
            "Acceptable values for 'log_out' parameter: 'file', 'both', 'console', None"
        )
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    # In case the same logger is called multiple times, don't attach new handlers
    if not logger.hasHandlers():
        if log_out.lower() in ("file", "both"):
            filename = Path(
                f"{log_path}/{name}{'_' + time.strftime('%Y-%m-%d_%H-%M-%S') if timedatelog else ''}.log"
            )
            filehandler = logging.FileHandler(filename, mode=mode)
            filehandler.setFormatter(
                logging.Formatter("%(asctime)s %(name)s %(levelname)-10s %(message)s")
            )
            logger.addHandler(filehandler)

        if log_out.lower() in ("console", "both"):
            logger.addHandler(logging.StreamHandler(sys.stdout))

    # this part of the code has issues with the handler lock. May fix one day.
    # # In case the same logger is called multiple times, don't attach new handlers
    # if not logger.hasHandlers():
    #     if log_out.lower() in ("file", "both"):
    #         filename = Path(
    #             f"{log_path}/{name}{'_' + time.strftime('%Y-%m-%d_%H-%M-%S') if timedatelog else ''}.log"
    #         )
    #         logger.addHandler(CustomFileHandler(filename=filename, mode=mode))
    #
    #     if log_out.lower() in ("console", "both"):
    #         logger.addHandler(CustomStreamHandler(sys.stdout))

    return logger


class CustomFileHandler(logging.FileHandler):
    def __init__(self, filename: Path, mode: str = "w"):
        super(CustomFileHandler, self).__init__(filename, mode=mode)
        self.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)-10s %(message)s"))

    def emit(self, record):
        # add prefix to multiline log messages
        messages = record.msg.split("\n")
        for message in messages:
            record.msg = message
            super(CustomFileHandler, self).emit(record)


class CustomStreamHandler(logging.StreamHandler):
    def __init__(self, stream=sys.stdout):
        super(CustomStreamHandler, self).__init__(stream)

    def emit(self, record):
        # add prefix to multiline log messages
        messages = record.msg.split("\n")
        for message in messages:
            record.msg = message
            super(CustomStreamHandler, self).emit(record)
