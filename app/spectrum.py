""" Gamma spectrum preprocessing """


import datetime
import json
import xmltodict
from typing import List
import numpy as np
from models import Spectrum
from scipy.signal import savgol_filter


class SpectrumPreprocessing:
    """ Gamma Spectrum Preprocessing (tested on spectra from Radiacode 103x devices) """

    @staticmethod
    def convert_to_features(spectrum: Spectrum, isotopes: List) -> np.array:
        """ Convert the spectrum to the list of features for prediction """
        sp_norm = SpectrumPreprocessing._normalize(spectrum)
        energies = [energy for _, energy in isotopes]
        channels = [SpectrumPreprocessing.energy_to_channel(spectrum, energy) for energy in energies]
        return np.array([sp_norm.counts[ch] for ch in channels])

    @staticmethod
    def _normalize(spectrum: Spectrum) -> Spectrum:
        """ Normalize data to the same count values """
        # Smooth data
        # counts = np.array(spectrum.counts) #.astype(np.float64)
        counts = SpectrumPreprocessing._smooth_data(spectrum.counts)

        # Normalize to 0..1
        val_norm = counts.max()
        return Spectrum(
            duration=spectrum.duration,
            a0 = spectrum.a0,
            a1 = spectrum.a1,
            a2 = spectrum.a2,
            counts = counts/val_norm
        )
      
    @staticmethod
    def _smooth_data(data: List | np.array) -> np.array:
        """ Apply 1D smoothing filter to the data array """
        window_size = 20
        data_out = savgol_filter(
            data,
            window_length=window_size,
            polyorder=2,
        )
        return np.clip(data_out, a_min=0, a_max=None)
    
    @staticmethod
    def get_counts(spectrum: Spectrum) -> List:
        """ Get all count values 'as is' """
        return spectrum.counts
    
    @staticmethod
    def get_channels(spectrum: Spectrum) -> List:
        """ Get all channel energies in KeV """
        channels = list(range(0, 1024))
        return [SpectrumPreprocessing.channel_to_energy(spectrum, ch) for ch in channels]
    
    @staticmethod
    def get_duration_sec(spectrum: Spectrum) -> int:
        """ Get spectrum duration in seconds """
        return int(spectrum.duration.total_seconds())
    
    @staticmethod
    def channel_to_energy(spectrum: Spectrum, ch: int) -> float:
        """ Convert channel number to the energy level """
        return spectrum.a0 + spectrum.a1 * ch + spectrum.a2 * ch**2
    
    @staticmethod
    def energy_to_channel(spectrum: Spectrum, e: float):
        """ Convert energy to the channel number (inverse E = a0 + a1*C + a2 C^2) """
        c = spectrum.a0 - e
        return int(
            (np.sqrt(spectrum.a1**2 - 4 * spectrum.a2 * c) - spectrum.a1) / (2 * spectrum.a2)
        )
    
    @staticmethod
    def create_empty() -> Spectrum:
        """ Create empty spectrum placeholder """
        return Spectrum(
            duration=0,
            a0 = 0,
            a1 = 0,
            a2 = 0,
            counts = [0]*1024
        )
    
    @staticmethod
    def load_from_file(filename: str) -> Spectrum:
        """ Load spectrum from a file """
        with open(filename) as f_in:
            data = json.load(f_in)
            return Spectrum(
                a0=data["a0"], a1=data["a1"], a2=data["a2"],
                counts=data["counts"],
                duration=datetime.timedelta(seconds=data["duration"]),
            )
        
    @staticmethod
    def to_string(sp: Spectrum) -> str:
        """ Convert spectrum to string """
        counts_str = ",".join([str(cnt) for cnt in sp.counts])
        return f"{int(sp.duration.total_seconds())};{sp.a0};{sp.a1};{sp.a2};{counts_str}"

        
    @staticmethod 
    def load_from_xml_file(file_path: str) -> Spectrum:
        """ Load spectrum from a Radiacode Android app file """
        with open(file_path) as f_in:
            return SpectrumPreprocessing.load_from_xml(f_in.read())


    @staticmethod 
    def load_from_xml(xml_data: str) -> Spectrum:
        """ Load spectrum from a Radiacode Android app file """
        doc = xmltodict.parse(xml_data)
        result = doc["ResultDataFile"]["ResultDataList"]["ResultData"]
        spectrum = result["EnergySpectrum"]
        calibration = spectrum["EnergyCalibration"]["Coefficients"]["Coefficient"]
        a0, a1, a2 = float(calibration[0]), float(calibration[1]), float(calibration[2])
        duration = int(spectrum["MeasurementTime"])
        data = spectrum["Spectrum"]["DataPoint"]
        return Spectrum(
            a0=a0, a1=a1, a2=a2,
            counts=[int(x) for x in data],
            duration=datetime.timedelta(seconds=duration),
        )
