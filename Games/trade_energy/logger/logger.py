import logging

def setup_logger():
    logging.basicConfig(filename='energy_model.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

def log_info(message):
    logging.info(message)

def log_error(error):
    logging.error(error)