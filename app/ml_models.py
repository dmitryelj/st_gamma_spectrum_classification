""" ML model for Radiacode gamma spectrum classification """


import logging
import json
import numpy as np
import os
from typing import List
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from spectrum import SpectrumPreprocessing
from models import SpectrumData


class IsotopesClassificationModel:
    """ Gamma Spectrum Classification Model """

    VERSION = "V1"
    MIN_SPECTRUM_DURATION_SEC = 30

    def __init__(self):
        """ Load models """
        path = self._get_models_path()
        self._classifier = self._load_model(path + "/XGBClassifier.json")
        self._isotopes = self._load_isotopes(path + "/isotopes.json")
        self._labels_encoder = self._load_labels_encoder(path + "/LabelEncoder.npy")
        logging.debug("ML model loaded")

    def predict(self, spectrum: SpectrumData) -> str:
        """ Predict the isotope """
        if spectrum.get_duration() < self.MIN_SPECTRUM_DURATION_SEC:
            return "Not Enough Data"

        features = SpectrumPreprocessing.convert_to_features(spectrum.get_data(), self._isotopes)
        preds = self._classifier.predict([features])
        preds = self._labels_encoder.inverse_transform(preds)
        return preds[0]

    @staticmethod
    def _load_model(filename: str) -> XGBClassifier:
        """ Load model from file """
        bst = XGBClassifier()
        bst.load_model(filename)
        return bst

    @staticmethod
    def _load_isotopes(filename: str) -> List:
        with open(filename, "r") as f_in:
            return json.load(f_in)
        
    @staticmethod
    def _load_labels_encoder(filename: str) -> LabelEncoder:
        le = LabelEncoder()
        le.classes_ = np.load(filename)
        return le

    @staticmethod
    def _get_models_path() -> str:
        """ Get path to models """
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return parent_dir + f"/models/{IsotopesClassificationModel.VERSION}"
