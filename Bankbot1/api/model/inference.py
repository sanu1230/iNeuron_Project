import logging
import numpy as np
import pandas as pd
import json
from api.config import get_logger

_logger = get_logger(logger_name=__name__)

from keras.models import load_model

from api.model.helper import *


def predict(input_data) -> list:
    """Make a prediction using the saved model"""
    print(input_data)
    input_data = json.loads(input_data)  # Comment for POSTMAN
    type, data = input_data['type'], input_data['data']

    _logger.info('Inside Predict')

    if type == 'conv':
        model = load_model(r"api\model\chatbot_model.h5")
        result = predict_class(data, model)
        print(data, result)
        _logger.info(data, result)
    return result