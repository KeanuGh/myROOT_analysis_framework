import logging
import sys
import time


def get_logger(
    log_level: int = 10,
    log_out: str = "console",
    name: str = "log",
    timedatelog: bool = False,
    log_dir: str = None,
    log_file: str = None,
    mode: str = "w",
) -> logging.Logger:
    """
    Generate logger object

    :param name: Name of logger
    :param log_level: Log level
    :param log_out: Whether to output log to 'file', 'console' or 'both'
    :param timedatelog: Whether to append datetime to log filename
    :param log_dir: Directory to save log file to if log_out is 'file' or 'both'. Ignored otherwise.
                    Pass either this or log_file
    :param log_file: File to log to if log_out is 'file' or 'both'. Ignored otherwise.
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
            if (log_dir and log_file) or (log_dir == log_file is None):
                raise ValueError("Pass either 'log_dir' or 'logfile'")

            filename = (
                log_file
                if log_file
                else f"{log_dir}/{name}{'_' + time.strftime('%Y-%m-%d_%H-%M-%S') if timedatelog else ''}.log"
            )
            filehandler = logging.FileHandler(filename, mode=mode)
            filehandler.setFormatter(
                logging.Formatter("%(asctime)s %(name)s %(levelname)-10s %(message)s")
            )
            logger.addHandler(filehandler)

        if log_out.lower() in ("console", "both"):
            logger.addHandler(logging.StreamHandler(sys.stdout))

    return logger
