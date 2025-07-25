""" Data models """


from dataclasses import dataclass
import datetime


@dataclass
class Spectrum:
    """ Radiation spectrum data """

    duration: datetime.timedelta
    a0: float
    a1: float
    a2: float
    counts: list[int]


@dataclass
class SpectrumData:
    """ Spectrum data wrapper """
    spectrum: Spectrum

    def get_duration(self) -> float:
        """ Get spectrum duration """
        return self.spectrum.duration.total_seconds()
    
    def get_data(self) -> Spectrum:
        """ Get spectrum data """
        return self.spectrum
