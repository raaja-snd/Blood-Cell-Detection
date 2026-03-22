
import logging
from pathlib import Path
from datetime import datetime

class AppLogger:

    _logger = None

    @staticmethod
    def get_logger(name):
        """
        Configure and creates a logger to log system level logs.
        The logs are stored in a timestamped .log file under <project_root>/logs/ folder.
        Ex. /logs/2025-10-5 13-12-22/2025-10-5 13-12-22.logs
        Args:
            name(str): Name to be used by the logger
        Returns:
            logging.logger: A configured logging instance 
        """

        #If logger exists return the existing logger
        if AppLogger._logger is not None:
            return AppLogger._logger

        filename = datetime.now().strftime(format='%Y-%m-%d %H-%M-%S')
        log_file = Path(__file__).resolve().parents[2]/'logs'/f'{filename}'/f'{filename}.log'
        if not log_file.parent.exists():
            log_file.parent.mkdir(parents=True,exist_ok=True)
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler(str(log_file),encoding='utf-8',mode='w')
        formatter = logging.Formatter('%(asctime)s %(levelname)s : %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.debug('Initialized Logger')
        AppLogger._logger = logger
        return logger



logger = AppLogger.get_logger('BCD')

