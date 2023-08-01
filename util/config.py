# -*- coding: utf-8 -*-

'''This module allows to configure the application behavior'''

import logging
LOGGING_LEVEL = logging.DEBUG
LOG_DIR = "./log2"
#-------------------------------------------------------------------------------
#  Logger configuration
#-------------------------------------------------------------------------------

formatter = logging.Formatter("%(asctime)s -- %(name)s -- %(levelname)s -- %(message)s")
#logger = None
def _create_all_handlers(logdir) :
   log_dir = logdir#LOG_DIR+"/"+logdir
   try :
      handler_debug   = logging.FileHandler(log_dir+"/debug.log", mode="a", encoding="utf-8")
      handler_info    = logging.FileHandler(log_dir+"/info.log", mode="a", encoding="utf-8")
      handler_warning = logging.FileHandler(log_dir+"/warning.log", mode="a", encoding="utf-8")
      handler_error   = logging.FileHandler(log_dir+"/error.log", mode="a", encoding="utf-8")
      handler_critic  = logging.FileHandler(log_dir+"/critic.log", mode="a", encoding="utf-8")
      handler_fatal   = logging.FileHandler(log_dir+"/fatal.log", mode="a", encoding="utf-8")

      handler_debug.setFormatter(formatter)
      handler_info.setFormatter(formatter)
      handler_warning.setFormatter(formatter)
      handler_error.setFormatter(formatter)
      handler_critic.setFormatter(formatter)
      handler_fatal.setFormatter(formatter)

      handler_debug.setLevel(logging.DEBUG)
      handler_info.setLevel(logging.INFO)
      handler_warning.setLevel(logging.WARNING)
      handler_error.setLevel(logging.ERROR)
      handler_critic.setLevel(logging.CRITICAL)
      handler_fatal.setLevel(logging.FATAL)
      '''
      logger.debug('Debug')
      logger.info('Info')
      logger.warning('Warning')
      logger.error('Error')
      '''
      print("Directory {} all handlers created!".format(log_dir))
      return [handler_debug, handler_info, handler_warning, handler_error, handler_critic, handler_fatal]
   except Exception as exception :
      logging.error("_create_all_handlers: {}".format(exception))

def get_logger(module_name, logdir):
   list_handler = _create_all_handlers(logdir)
   try :
      logger = logging.getLogger(module_name)
      logger.setLevel(LOGGING_LEVEL)
      for handler in list_handler :
         try:
            logger.addHandler(handler)
         except Exception as exception :
            logging.error(exception)
   except Exception as exception :
      logging.error(exception)
   return logger

def close_all_handlers(logger) :
   try :
      for handler in logger.handlers :
         handler.close()
   except Exception as exception :
      print("Files could not been closed: {}".format(exception))
#-------------------------------------------------------------------------------
