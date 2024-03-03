# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from dotenv import dotenv_values
from . import paths as PATHS


"""
Environment file
"""
ENV = dotenv_values(os.path.join(PATHS.PACKAGE_PATH, ".env"))


"""
Logger
"""


class LOGGER_REPLACEMENT(object):
    """
    Logger replacement class.
    """

    def debug(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[DEBUG] {text}")

    def info(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[INFO] {text}")

    def warning(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[WARNING] {text}")

    def warn(self, text: str) -> None:
        """
        Method replacement for logging.
        :param text: Text to log.
        """
        print(f"[WARNING] {text}")


LOGGER = LOGGER_REPLACEMENT()


"""
Project information
"""
PROJECT_NAME = "Basic ML model backend"
PROJECT_DESCRIPTION = "Tool for managing machine learning models."
PROJECT_VERSION = "v0.2"


"""
Network addresses
"""
BACKEND_HOST = ENV.get("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = ENV.get("BACKEND_PORT", "7861")
FRONTEND_HOST = ENV.get("FRONTEND_HOST", "127.0.0.1")
FRONTEND_PORT = ENV.get("FRONTEND_PORT", "8868")


"""
Others
"""
TIMESTAMP_FORMAT = "%m/%d/%Y, %H:%M:%S"
FILE_UPLOAD_CHUNK_SIZE = 1024*1024
FRONTEND_GRAPH_SIZE = [1080, 680]