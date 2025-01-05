#!/usr/bin/env python
# coding: utf-8

import datetime

LogLevels = {"CRITICAL": 0, "ERROR": 1, "WARNING": 2, "INFO": 3, "DEBUG": 4}


class TinyLogger:
    def __init__(self, log_level="ERROR", log_file=None, log_to_console=True):
        self.log_level = "ERROR"
        if log_level.upper() in LogLevels:
            self.log_level = log_level.upper()

        self.log_file = log_file
        self.log_to_console = log_to_console

    def set_level(self, log_level="ERROR"):
        self.log_level = "ERROR"
        if log_level.upper() in LogLevels:
            self.log_level = log_level.upper()

    def _log(self, level, message):
        if LogLevels[level.upper()] <= LogLevels[self.log_level]:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"{timestamp} - {level} - {message}"

            if self.log_file:
                with open(self.log_file, "a") as file:
                    file.write(log_message + "\n")

            if self.log_to_console:
                print(log_message)

    def debug(self, message):
        self._log("DEBUG", message)

    def info(self, message):
        self._log("INFO", message)

    def warning(self, message):
        self._log("WARNING", message)

    def error(self, message):
        self._log("ERROR", message)

    def critical(self, message):
        self._log("CRITICAL", message)
