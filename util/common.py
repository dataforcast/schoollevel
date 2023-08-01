# -*- coding: utf-8 -*-
import os
from pathlib import Path
import shutil
import logging
from util import config as config_util


# logger = config_util.logger

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
def yield_pathdir_from_pathfile(pathfile: str) -> str:
    """Extract the path dir from a path file"""
    return Path(pathfile).parent.name

# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
class MyLogger():
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def debug(self, message):
        try:
            self.logger.debug(message)
        except Exception as exception:
            logging.warning(message)

    def info(self, message):
        try:
            self.logger.info(message)
        except Exception as exception:
            logging.warning(message)
            
    def warning(self, message):
        try:
            self.logger.warning(message)
        except Exception as exception:
            logging.warning(message)
    def error(self, message):
        try:
            self.logger.error(message)
        except Exception as exception:
            logging.warning(message)
    def critic(self, message):
        try:
            self.logger.critic(message)
        except Exception as exception:
            logging.warning(message)
    def fatal(self, message):
        try:
            self.logger.fatal(message)
        except Exception as exception:
            logging.warning(message)
# -------------------------------------------------------------------------------
    
# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------

def halt(logger, exception: str = ''):
    """When invoked, the program is halted.
    When running in a notebook, then os._exit(-1) is invoked,
    otherwise, sys.exit(-1) is invoked.
    """
    halt_message = '{} {}'.format(exception, "Program halt!")
    logging.fatal(halt_message)
    try:
        if logger is not None:
            logger.fatal(halt_message)
        os._exit(-1)
    except Exception as exception:
        os._exit(-1)
# -------------------------------------------------------------------------------


# -------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------
class Common():
    """This class contains methods and properties that are common to all
    classes.
    """
    LOGGER_PROMPT = "\n-->{}"
    SETTER_MESSAGE_ERROR = " Forbidden operation: update"

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def __init__(self, logger=None):
        """Initialise common attributes and methods that will be
        inherited in child classes instances.
        If log directory exists, then it is deleted and created.
        """
        self.is_assigned_logger = False
        if logger is None:
            self.logger = None
            self.log_dir = config_util.LOG_DIR + '/' + "{}_{}".format(config_util.LOG_DIR, type(self).__name__)
            self.update_log()

            logger = config_util.get_logger(type(self).__name__, self.log_dir)
            self.logger = MyLogger(logger)
        else:
            self.logger =  MyLogger(logger)
            self.is_assigned_logger = True

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def __del__(self):
        """Clean all ressources when class is garbage-collected"""
        if self.is_assigned_logger:
            self.logger.logger = None
        else:
            pass
        self.clean_log()

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def create_log(self):
        try:
            os.makedirs(self.log_dir)
            logging.info("Directory {} created!".format(self.log_dir))
        except Exception as exception:
            logging.error("create_log: {}".format(exception))

    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def clean_log(self):
        if os.path.isdir(self.log_dir):
            try:
                if self.logger is not None:
                    if self.logger.logger is not None:
                        config_util.close_all_handlers(self.logger.logger)
                        del(self.logger.logger)
                else:
                    pass
                shutil.rmtree(self.log_dir)
                logging.info("Directory {} cleaned!".format(self.log_dir))
            except Exception as exception:
                logging.error("{}".format(exception))
        else:
            pass

    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    #
    # --------------------------------------------------------------------------
    def update_log(self):
        self.clean_log()
        self.create_log()

    # --------------------------------------------------------------------------

# -------------------------------------------------------------------------------
