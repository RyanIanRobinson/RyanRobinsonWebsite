import logging

def setup_logger():
    logging.basicConfig(filename='model_training.log',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

def log_training_info(message):
    logging.info(message)

def log_error(error):
    logging.error(error)
