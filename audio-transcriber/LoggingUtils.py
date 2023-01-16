import os
import logging
from logging.handlers import RotatingFileHandler

# Logging Configs
logSize = 10
backupCount = 30
delay = 0

LEVEL = 20

'''
logging.CRITICAL = 50
logging.FATAL = CRITICAL
logging.ERROR = 40
logging.WARNING = 30
logging.WARN = WARNING
logging.INFO = 20
logging.DEBUG = 10
logging.NOTSET = 0
'''

class LogFacilitator:
    def __init__(self, logFilePath, loggerName, level=LEVEL):
        self.__logFilePath = self._initLogFilePath(logFilePath)
        self.__level = level
        self.__logFormat = logging.Formatter(
            "[%(levelname)s][%(asctime)s][%(filename)s:%(lineno)d]:%(message)s"
        )
        self.__rotatingFileHandler = self._initRotatingFileHandler()
        self.logger = self._initLogger(loggerName)

    def _initLogFilePath(self, path):
        currentDirectory = os.getcwd()
        folder = "logs"
        directory = os.path.join(currentDirectory, folder)
        fileName = path + ".log"
        path = os.path.join(directory, fileName)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        if not os.path.isfile(path):
            with open(path, "a") as log:
                log.write("New log file started.\n")
        return path

    def _initRotatingFileHandler(self):
        rotatingFileHandler = RotatingFileHandler(
            self.__logFilePath,
            mode="a",
            maxBytes=logSize * 1024 * 1024,
            backupCount=backupCount,
            encoding=None,
            delay=delay,
        )
        rotatingFileHandler.setFormatter(self.__logFormat)
        rotatingFileHandler.setLevel(self.__level)
        return rotatingFileHandler

    def _initLogger(self, loggerName):
        logger = logging.getLogger(loggerName)
        logger.setLevel(self.__level)
        logger.addHandler(self.__rotatingFileHandler)
        logger.propagate = False
        return logger


MainLogger = LogFacilitator("Main", "main.py")